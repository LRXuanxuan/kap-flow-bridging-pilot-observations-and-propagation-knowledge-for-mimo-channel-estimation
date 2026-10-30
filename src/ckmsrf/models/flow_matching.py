"""CKM-conditioned velocity field for additive Flow Matching."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .unet import ConditionalGMUNet


class ConditionalFlowMatchingUNet(ConditionalGMUNet):
    """Predict the velocity of the additive path ``x_sigma=x0+sigma*epsilon``.

    This class deliberately has the same backbone, condition encoder and
    two-channel output-head capacity as :class:`ConditionalConsistencyUNet`.
    It differs only in output semantics: ``forward`` returns the path velocity
    (the artificial perturbation ``epsilon``), not a clean-channel estimate.

    The input and noise-coordinate preconditioning match the CM/Diffusion
    backbone so a controlled experiment changes the training target without
    silently changing network width or CKM fusion.
    """

    output_parameterization = "velocity"

    def __init__(
        self,
        *args: Any,
        sigma_min: float = 2.0e-3,
        sigma_data: float = 0.5,
        zero_init_output: bool = True,
        **kwargs: Any,
    ) -> None:
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be positive")
        if sigma_data <= 0.0:
            raise ValueError("sigma_data must be positive")

        super().__init__(*args, **kwargs)
        self.sigma_min = float(sigma_min)
        self.sigma_data = float(sigma_data)

        head = nn.Conv2d(
            self.final_conv.in_channels,
            self.out_channels,
            kernel_size=self.final_conv.kernel_size,
            stride=self.final_conv.stride,
            padding=self.final_conv.padding,
        )
        if zero_init_output:
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        self.final_conv = head
        del self.logstd_mlp

    def _format_gm_output(
        self,
        raw: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        del embedding
        return raw

    def _prepare_sigma(
        self, sigma: torch.Tensor | float, x: torch.Tensor
    ) -> torch.Tensor:
        value = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if value.numel() == 1:
            value = value.reshape(1).expand(x.shape[0])
        else:
            value = value.reshape(-1)
        if value.shape[0] != x.shape[0]:
            raise ValueError(
                f"Sigma batch {value.shape[0]} does not match input batch {x.shape[0]}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("sigma must contain only finite values")
        if bool((value < self.sigma_min).any()):
            raise ValueError(
                f"sigma must be greater than or equal to sigma_min={self.sigma_min}"
            )
        return value

    def input_preconditioning(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shared EDM input scale and log-noise coordinate."""

        sigma_data = torch.as_tensor(
            self.sigma_data, device=sigma.device, dtype=sigma.dtype
        )
        c_in = torch.rsqrt(sigma.square() + sigma_data.square())
        c_noise = sigma.log() * 0.25
        return c_in, c_noise

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float,
        condition: Mapping[str, Any] | None = None,
        *,
        context: Mapping[str, Any] | None = None,
        cond_drop_mask: torch.Tensor | None = None,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.init_conv.in_channels:
            raise ValueError(
                f"Expected [B,{self.init_conv.in_channels},Nr,Nt], got {tuple(x.shape)}"
            )
        sigma_batch = self._prepare_sigma(sigma, x)
        c_in, c_noise = self.input_preconditioning(sigma_batch)
        expand = (-1,) + (1,) * (x.ndim - 1)
        return super().forward(
            x * c_in.view(expand),
            c_noise / 1000.0,
            condition=condition,
            context=context,
            cond_drop_mask=cond_drop_mask,
            force_unconditional=force_unconditional,
        )

    def denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float,
        condition: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Convert the learned velocity into a one-step clean estimate."""

        sigma_batch = self._prepare_sigma(sigma, x)
        expand = (-1,) + (1,) * (x.ndim - 1)
        velocity = self.forward(x, sigma_batch, condition=condition, **kwargs)
        return x - sigma_batch.view(expand) * velocity


__all__ = ["ConditionalFlowMatchingUNet"]
