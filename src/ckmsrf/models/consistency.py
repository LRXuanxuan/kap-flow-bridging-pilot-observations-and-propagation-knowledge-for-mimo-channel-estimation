"""CKM-conditioned consistency model built on the existing channel U-Net.

``sigma`` is the standard deviation of the artificial corruption used by the
generative model.  It is deliberately kept separate from the physical
snapshot coordinate, which remains available as ``condition["time"]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .unet import ConditionalGMUNet


class ConditionalConsistencyUNet(ConditionalGMUNet):
    """Direct clean-channel predictor with consistency preconditioning.

    The encoder, CKM condition encoder, FiLM-like residual blocks and decoder
    are inherited from :class:`ConditionalGMUNet`.  Only the Gaussian-mixture
    head is replaced by a two-channel residual head.  The returned tensor is
    always a clean-channel estimate with shape ``[B, 2, Nr, Nt]``.

    The parameterization follows the consistency-model boundary construction

    ``f(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma) * x)``.

    At ``sigma == sigma_min``, ``c_skip=1`` and ``c_out=0``.  A final explicit
    selection makes the identity boundary exact, including under mixed
    precision.
    """

    output_parameterization = "clean"

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

        # Keep all backbone parameter names compatible with ConditionalGMUNet;
        # only the task-specific output head changes shape.
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

        # The mixture-scale MLP belongs to the old GM head and is not used by a
        # direct consistency predictor.
        del self.logstd_mlp

    def _format_gm_output(
        self,
        raw: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        del embedding
        return raw

    def _prepare_sigma(self, sigma: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
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
            raise ValueError(f"sigma must be greater than or equal to sigma_min={self.sigma_min}")
        return value

    def preconditioning(
        self,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``c_skip``, ``c_out``, ``c_in`` and ``c_noise``.

        ``c_noise`` is the log-noise coordinate supplied to the inherited time
        embedding.  It is unrelated to the physical snapshot-time condition.
        """

        sigma_data = torch.as_tensor(self.sigma_data, device=sigma.device, dtype=sigma.dtype)
        sigma_min = torch.as_tensor(self.sigma_min, device=sigma.device, dtype=sigma.dtype)
        offset = (sigma - sigma_min).clamp_min(0.0)
        sigma_sq = sigma.square()
        data_sq = sigma_data.square()
        c_skip = data_sq / (offset.square() + data_sq)
        c_out = sigma_data * offset / torch.sqrt(sigma_sq + data_sq)
        c_in = torch.rsqrt(sigma_sq + data_sq)
        c_noise = sigma.log() * 0.25
        return c_skip, c_out, c_in, c_noise

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
        c_skip, c_out, c_in, c_noise = self.preconditioning(sigma_batch)
        expand = (-1,) + (1,) * (x.ndim - 1)

        # ConditionalGMUNet multiplies its second argument by 1000 before the
        # inherited embedding.  Divide here so the embedding receives exactly
        # the standard CM log-noise coordinate c_noise.
        residual = super().forward(
            x * c_in.view(expand),
            c_noise / 1000.0,
            condition=condition,
            context=context,
            cond_drop_mask=cond_drop_mask,
            force_unconditional=force_unconditional,
        )
        estimate = c_skip.view(expand) * x + c_out.view(expand) * residual

        # This explicit branch is intentional: it gives a bitwise identity at
        # the boundary even when autocast rounds the scalar coefficients.
        boundary = (sigma_batch == self.sigma_min).view(expand)
        return torch.where(boundary, x, estimate)


__all__ = ["ConditionalConsistencyUNet"]
