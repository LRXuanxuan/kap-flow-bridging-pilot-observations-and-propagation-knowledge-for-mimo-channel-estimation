"""Legacy-compatible 2-D Gaussian-mixture U-Net and CKM-conditioned variant."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .condition import CKMConditionEncoder


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        if half_dim <= 1:
            emb = x[:, None]
        else:
            scale = math.log(10000) / (half_dim - 1)
            frequencies = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -scale)
            emb = x[:, None] * frequencies[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return F.pad(emb, (0, max(self.dim - emb.shape[-1], 0)))[:, : self.dim]


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(math.gcd(groups, dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1.0) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int | None = None, groups: int = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            scale_shift = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class GMUNet(nn.Module):
    """The original SRF 2-D U-Net with a Gaussian-mixture output head.

    Module names and tensor shapes intentionally match ``MIMO_gm2.models.UNet``
    so legacy state dictionaries can be loaded without renaming keys.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        model_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        gm_num_components: int = 4,
        gm_min_logstd: float = -3.5,
        gm_max_logstd: float = 0.5,
        gm_logstd_hidden: int | None = None,
        gm_logstd_layers: int = 2,
        groups: int = 8,
        *,
        dim: int | None = None,
        dim_mults: tuple[int, ...] | list[int] | None = None,
        num_components: int | None = None,
    ):
        super().__init__()
        model_channels = int(dim if dim is not None else model_channels)
        channel_mult = tuple(dim_mults) if dim_mults is not None else tuple(channel_mult)
        gm_num_components = int(num_components if num_components is not None else gm_num_components)
        self.out_channels = int(out_channels)
        self.gm_num_components = gm_num_components
        self.gm_min_logstd = float(gm_min_logstd)
        self.gm_max_logstd = float(gm_max_logstd)
        self.time_dim = model_channels * 4
        self.gradient_checkpointing = False

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        ch = model_channels
        ds_chs = [ch]
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResnetBlock(ch, out_ch, time_emb_dim=self.time_dim, groups=groups))
                ch = out_ch
                ds_chs.append(ch)
            if level != len(channel_mult) - 1:
                self.downs.append(Downsample(ch))
                ds_chs.append(ch)

        self.mid_block1 = ResnetBlock(ch, ch, time_emb_dim=self.time_dim, groups=groups)
        self.mid_block2 = ResnetBlock(ch, ch, time_emb_dim=self.time_dim, groups=groups)

        self.ups = nn.ModuleList([])
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(
                    ResnetBlock(ch + ds_chs.pop(), out_ch, time_emb_dim=self.time_dim, groups=groups)
                )
                ch = out_ch
            if level != len(channel_mult) - 1:
                self.ups.append(Upsample(ch))

        self.final_res_block = ResnetBlock(ch, ch, time_emb_dim=self.time_dim, groups=groups)
        head_channels = gm_num_components * self.out_channels + gm_num_components
        self.final_conv = nn.Conv2d(ch, head_channels, 3, padding=1)

        hidden = gm_logstd_hidden or self.time_dim
        logstd_layers: list[nn.Module] = []
        in_dim = self.time_dim
        for _ in range(max(gm_logstd_layers - 1, 0)):
            logstd_layers.extend((nn.SiLU(), nn.Linear(in_dim, hidden)))
            in_dim = hidden
        logstd_layers.extend((nn.SiLU(), nn.Linear(in_dim, 1)))
        self.logstd_mlp = nn.Sequential(*logstd_layers)

    def _bound_logstd(self, raw_logstd: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(raw_logstd)
        logstd = self.gm_min_logstd + 0.5 * (bounded + 1.0) * (
            self.gm_max_logstd - self.gm_min_logstd
        )
        return logstd.view(raw_logstd.shape[0], 1, 1, 1, 1)

    def _format_gm_output(self, raw: torch.Tensor, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, height, width = raw.shape
        means_raw, logits_raw = torch.split(
            raw,
            [self.gm_num_components * self.out_channels, self.gm_num_components],
            dim=1,
        )
        means = means_raw.view(batch, self.gm_num_components, self.out_channels, height, width)
        logweights = logits_raw.view(batch, self.gm_num_components, 1, height, width).log_softmax(dim=1)
        # Kept detached to reproduce the legacy GM head exactly.
        logstds = self._bound_logstd(self.logstd_mlp(embedding.detach()))
        return {"means": means, "logstds": logstds, "logweights": logweights}

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Trade extra forward computation for lower activation memory."""

        self.gradient_checkpointing = bool(enabled)

    def _run_resnet(
        self, module: ResnetBlock, x: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, embedding, use_reentrant=False)
        return module(x, embedding)

    def _forward_with_embedding(self, x: torch.Tensor, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.init_conv(x)
        skips = [x]
        for module in self.downs:
            if isinstance(module, ResnetBlock):
                x = self._run_resnet(module, x, embedding)
                skips.append(x)
            else:
                x = module(x)
                skips.append(x)
        x = self._run_resnet(self.mid_block1, x, embedding)
        x = self._run_resnet(self.mid_block2, x, embedding)
        for module in self.ups:
            if isinstance(module, ResnetBlock):
                skip = skips.pop()
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
                x = self._run_resnet(module, torch.cat((x, skip), dim=1), embedding)
            else:
                x = module(x)
        x = self._run_resnet(self.final_res_block, x, embedding)
        return self._format_gm_output(self.final_conv(x), embedding)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        t = t.reshape(-1).to(device=x.device, dtype=x.dtype)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"Time batch {t.shape[0]} does not match input batch {x.shape[0]}")
        return self._forward_with_embedding(x, self.time_mlp(t * 1000.0))


class ConditionalGMUNet(GMUNet):
    """Legacy GM U-Net with zero-initialized continuous CKM injection."""

    def __init__(
        self,
        *args: Any,
        condition_encoder: CKMConditionEncoder | None = None,
        condition_embed_dim: int = 256,
        cond_dropout: float = 0.1,
        condition: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        # ``condition={...}`` permits direct construction from the YAML block.
        condition_kwargs = dict(condition or {})
        condition_embed_dim = int(condition_kwargs.pop("embed_dim", condition_embed_dim))
        cond_dropout = float(
            condition_kwargs.pop("dropout_prob", condition_kwargs.pop("cond_dropout", cond_dropout))
        )
        self.force_unconditional = bool(
            condition_kwargs.pop("force_unconditional", False)
        )
        condition_kwargs.pop("enabled", None)
        super().__init__(*args, **kwargs)
        self.condition_encoder = condition_encoder or CKMConditionEncoder(
            embed_dim=condition_embed_dim,
            **condition_kwargs,
        )
        self.condition_projection = nn.Linear(condition_embed_dim, self.time_dim)
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        if not 0.0 <= cond_dropout <= 1.0:
            raise ValueError("cond_dropout must lie in [0, 1]")
        self.cond_dropout = cond_dropout

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Mapping[str, Any] | None = None,
        *,
        context: Mapping[str, Any] | None = None,
        cond_drop_mask: torch.Tensor | None = None,
        force_unconditional: bool = False,
    ) -> dict[str, torch.Tensor]:
        if condition is not None and context is not None:
            raise ValueError("Pass either condition or context, not both")
        condition = condition if condition is not None else context
        t = t.reshape(-1).to(device=x.device, dtype=x.dtype)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])
        time_embedding = self.time_mlp(t * 1000.0)
        condition_embedding = self.condition_encoder(
            condition,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        force_unconditional = bool(force_unconditional or self.force_unconditional)
        if force_unconditional:
            keep = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        elif cond_drop_mask is not None:
            drop = cond_drop_mask.to(device=x.device, dtype=torch.bool).reshape(x.shape[0], 1)
            keep = (~drop).to(dtype=x.dtype)
        elif self.training and self.cond_dropout > 0.0:
            keep = (torch.rand(x.shape[0], 1, device=x.device) >= self.cond_dropout).to(x.dtype)
        else:
            keep = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        embedding = time_embedding + self.condition_projection(condition_embedding * keep)
        return self._forward_with_embedding(x, embedding)


# Backwards-compatible class name used by the original scripts/checkpoints.
UNet = GMUNet
