"""CKM-conditioned velocity field for the straight latent-to-data flow path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .unet import ConditionalGMUNet


class ConditionalStraightFlowUNet(ConditionalGMUNet):
    r"""Predict velocity on ``x_t=(1-t)x_latent+t*x_data``.

    PnP-Flow and FlowADMM require this orientation of the path: ``t=0`` is
    the standard-normal latent distribution and ``t=1`` is the channel-data
    distribution.  The output is the straight-path velocity
    ``x_data-x_latent``.  This is intentionally a different semantic type
    from :class:`ConditionalFlowMatchingUNet`, whose additive/VE path is
    ``x_sigma=x_data+sigma*epsilon``.

    The backbone and CKM condition encoder remain capacity matched to the
    direct CM, Diffusion and additive-Flow priors.  No EDM input
    preconditioning is applied because the training path is the convex
    interpolation used by the PnP-Flow and FlowADMM papers.
    """

    output_parameterization = "straight_velocity"

    def __init__(
        self,
        *args: Any,
        zero_init_output: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
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

    @staticmethod
    def _prepare_time(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if value.numel() == 1:
            value = value.reshape(1).expand(x.shape[0])
        else:
            value = value.reshape(-1)
        if value.shape[0] != x.shape[0]:
            raise ValueError(
                f"Time batch {value.shape[0]} does not match input batch {x.shape[0]}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("t must contain only finite values")
        if bool(((value < 0.0) | (value > 1.0)).any()):
            raise ValueError("t must lie in [0, 1]")
        return value

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
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
        time = self._prepare_time(t, x)
        return super().forward(
            x,
            time,
            condition=condition,
            context=context,
            cond_drop_mask=cond_drop_mask,
            force_unconditional=force_unconditional,
        )

    def denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        condition: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Evaluate ``D_t(x)=x+(1-t)v_theta(x,t)``."""

        time = self._prepare_time(t, x)
        velocity = self.forward(x, time, condition=condition, **kwargs)
        expand = (-1,) + (1,) * (x.ndim - 1)
        return x + (1.0 - time).view(expand) * velocity


__all__ = ["ConditionalStraightFlowUNet"]
