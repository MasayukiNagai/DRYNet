from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def make_norm(norm_type: str | None, channels: int, num_groups: int = 8) -> nn.Module:
    norm_type = "none" if norm_type is None else norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "layer":
        return ChannelLayerNorm(channels)
    if norm_type == "group":
        groups = min(num_groups, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def make_dense_norm(norm_type: str | None, channels: int, num_groups: int = 8) -> nn.Module:
    norm_type = "none" if norm_type is None else norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "layer":
        return nn.LayerNorm(channels)
    if norm_type == "group":
        groups = min(num_groups, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported dense norm_type: {norm_type}")


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def same_padding(kernel_size: int, dilation: int = 1) -> tuple[int, int]:
    total = dilation * (kernel_size - 1)
    left = total // 2
    return left, total - left


def center_crop_1d(x: Tensor, target_len: int) -> Tensor:
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] < target_len:
        return F.interpolate(x, size=target_len, mode="nearest")
    start = (x.shape[-1] - target_len) // 2
    return x[..., start : start + target_len]


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class SameMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> Tensor:
        input_len = x.shape[-1]
        out_len = (input_len + self.stride - 1) // self.stride
        pad_total = max((out_len - 1) * self.stride + self.kernel_size - input_len, 0)
        left = pad_total // 2
        right = pad_total - left
        if pad_total > 0:
            x = F.pad(x, (left, right))
        return F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)


class StandardizedConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, dilation: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, padding=0, dilation=dilation, bias=True)
        self.scale = nn.Parameter(torch.ones(out_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        mean = weight.mean(dim=(1, 2), keepdim=True)
        var = weight.var(dim=(1, 2), unbiased=False, keepdim=True)
        fan_in = self.in_channels * self.kernel_size[0]
        min_var = torch.tensor(1e-4, device=weight.device, dtype=weight.dtype)
        scale = torch.rsqrt(torch.maximum(var * fan_in, min_var)) * self.scale
        weight = (weight - mean) * scale
        left, right = same_padding(self.kernel_size[0], self.dilation[0])
        x = F.pad(x, (left, right))
        return F.conv1d(x, weight, self.bias, self.stride, padding=0, dilation=self.dilation, groups=self.groups)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        norm_type: str = "batch",
        num_groups: int = 8,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = make_norm(norm_type, in_channels, num_groups)
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        if kernel_size == 1 and dilation == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.conv = StandardizedConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.dropout(self.activation(self.norm(x))))


class DnaEmbedder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dna_embedder_channels: int,
        *,
        norm_type: str,
        num_groups: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, dna_embedder_channels, kernel_size=15, padding="same")
        self.block = ConvBlock(
            dna_embedder_channels,
            dna_embedder_channels,
            kernel_size=5,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        return out + self.block(out)


class DownResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm_type: str,
        num_groups: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("DownResBlock out_channels must be >= in_channels.")
        self.out_channels = out_channels
        self.block1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=5,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.block2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=5,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.block1(x)
        pad_channels = self.out_channels - x.shape[1]
        if pad_channels > 0:
            x = F.pad(x, (0, 0, 0, pad_channels))
        out = out + x
        return out + self.block2(out)


class UpResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        norm_type: str,
        num_groups: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.conv_in = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=5,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.skip_proj = ConvBlock(
            skip_channels,
            out_channels,
            kernel_size=1,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.conv_out = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=5,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        out = self.conv_in(x)
        if x.shape[1] >= out.shape[1]:
            residual = x[:, : out.shape[1], :]
        else:
            residual = F.pad(x, (0, 0, 0, out.shape[1] - x.shape[1]))
        out = torch.repeat_interleave(out + residual, repeats=2, dim=2) * self.residual_scale
        if out.shape[-1] != skip.shape[-1]:
            target = max(out.shape[-1], skip.shape[-1])
            out = center_crop_1d(out, target)
            skip = center_crop_1d(skip, target)
        out = out + self.skip_proj(skip)
        return out + self.conv_out(out)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        dna_embedder_channels: int,
        encoder_channels: list[int],
        pool_size: int,
        norm_type: str,
        num_groups: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.embedder = DnaEmbedder(
            input_channels=input_channels,
            dna_embedder_channels=dna_embedder_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.pool = SameMaxPool1d(pool_size)
        self.down_blocks = nn.ModuleList()
        in_channels = dna_embedder_channels
        for out_channels in encoder_channels:
            self.down_blocks.append(
                DownResBlock(
                    in_channels,
                    out_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    activation=activation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        x = self.embedder(x)
        intermediates = {"bin_size_1": x}
        x = self.pool(x)
        bin_size = 2
        for block in self.down_blocks:
            x = block(x)
            intermediates[f"bin_size_{bin_size}"] = x
            x = self.pool(x)
            bin_size *= 2
        return x, intermediates


class SequenceDecoder(nn.Module):
    def __init__(
        self,
        *,
        dna_embedder_channels: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        norm_type: str,
        num_groups: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        full_skip_channels = list(reversed(encoder_channels)) + [dna_embedder_channels]
        if len(decoder_channels) != len(full_skip_channels):
            raise ValueError(
                "decoder_channels must contain one entry per decoder skip connection. "
                f"Expected {len(full_skip_channels)}, got {len(decoder_channels)}."
            )
        current_channels = encoder_channels[-1]
        self.bin_sizes = [2 ** len(encoder_channels), *[2 ** i for i in range(len(encoder_channels) - 1, -1, -1)]]
        self.up_blocks = nn.ModuleList()
        for skip_ch, out_ch in zip(full_skip_channels, decoder_channels):
            self.up_blocks.append(
                UpResBlock(
                    in_channels=current_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    activation=activation,
                    dropout=dropout,
                )
            )
            current_channels = out_ch

    def forward(self, x: Tensor, intermediates: dict[str, Tensor]) -> Tensor:
        for block, bin_size in zip(self.up_blocks, self.bin_sizes):
            x = block(x, intermediates[f"bin_size_{bin_size}"])
        return x


class ResidualConvUnit(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        norm_type: str,
        num_groups: int,
        activation: str,
        dropout: float,
        dilation: int = 1,
    ):
        super().__init__()
        self.block1 = ConvBlock(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.block2 = ConvBlock(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block2(self.block1(x))


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        mlp_ratio: float,
        norm_type: str,
        num_groups: int,
        activation: str,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        self.pre_norm = ChannelLayerNorm(channels) if norm_type == "layer" else make_norm(norm_type, channels, num_groups)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        hidden_dim = max(channels, int(channels * mlp_ratio))
        self.ffn_norm = ChannelLayerNorm(channels) if norm_type == "layer" else make_norm(norm_type, channels, num_groups)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.pre_norm(x).transpose(1, 2)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out.transpose(1, 2)
        z = self.ffn_norm(x).transpose(1, 2)
        return x + self.ffn(z).transpose(1, 2)


class HybridAttentionUnit(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        num_heads: int,
        mlp_ratio: float,
        norm_type: str,
        num_groups: int,
        activation: str,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.conv = ResidualConvUnit(
            channels,
            kernel_size=kernel_size,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
        )
        self.attn = SelfAttentionBlock(
            channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_type=norm_type,
            num_groups=num_groups,
            activation=activation,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.attn(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        bottleneck_cfg: dict[str, Any],
        fallback_norm_type: str,
        fallback_num_groups: int,
        fallback_activation: str,
    ):
        super().__init__()
        cfg = dict(bottleneck_cfg)
        norm_type = cfg.get("norm_type", fallback_norm_type)
        num_groups = int(cfg.get("num_groups", fallback_num_groups))
        activation = cfg.get("activation", fallback_activation)
        blocks: list[nn.Module] = []
        for _ in range(cfg["depth"]):
            if cfg["type"] == "residual_conv":
                blocks.append(
                    ResidualConvUnit(
                        channels,
                        kernel_size=cfg["kernel_size"],
                        norm_type=norm_type,
                        num_groups=num_groups,
                        activation=activation,
                        dropout=cfg["dropout"],
                    )
                )
            elif cfg["type"] == "hybrid_attention":
                blocks.append(
                    HybridAttentionUnit(
                        channels,
                        kernel_size=cfg["kernel_size"],
                        num_heads=cfg["num_heads"],
                        mlp_ratio=cfg["mlp_ratio"],
                        norm_type=norm_type,
                        num_groups=num_groups,
                        activation=activation,
                        dropout=cfg["dropout"],
                        attn_dropout=cfg["attn_dropout"],
                    )
                )
            else:
                raise ValueError(f"Unsupported bottleneck type: {cfg['type']}")
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class EmbeddingProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, norm_type: str, num_groups: int, activation: str, dropout: float):
        super().__init__()
        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = make_norm(norm_type, out_channels, num_groups)
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.norm(self.project(x))))


class IdentityProjector(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class ProfileHead(nn.Module):
    def __init__(self, input_channels: int, num_outputs: int, kernel_size: int, output_length: int):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, num_outputs, kernel_size=kernel_size)
        self.output_length = output_length

    def forward(self, x: Tensor) -> Tensor:
        return center_crop_1d(self.conv(x), self.output_length)


class YCountHead(nn.Module):
    def __init__(
        self,
        input_channels: int,
        *,
        conv_hidden_dims: list[int],
        mlp_hidden_dims: list[int],
        kernel_size: int,
        pool_size: int,
        adaptive_pool_size: int,
        dropout: float,
        norm_type: str,
        activation: str,
        mlp_norm_type: str = "layer",
        num_groups: int = 8,
    ):
        super().__init__()
        conv_layers: list[nn.Module] = []
        channels = input_channels
        for hidden in conv_hidden_dims:
            conv_layers.extend(
                [
                    nn.Conv1d(channels, hidden, kernel_size=kernel_size, padding="same", bias=False),
                    make_norm(norm_type, hidden, num_groups),
                    make_activation(activation),
                    nn.Dropout(dropout),
                    SameMaxPool1d(pool_size),
                ]
            )
            channels = hidden
        self.conv_stack = nn.Sequential(*conv_layers)
        self.adaptive_pool_size = adaptive_pool_size

        mlp_layers: list[nn.Module] = []
        mlp_input_dim = channels * adaptive_pool_size
        for hidden in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(mlp_input_dim, hidden),
                    make_dense_norm(mlp_norm_type, hidden, num_groups),
                    make_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            mlp_input_dim = hidden
        mlp_layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_stack(x)
        x = F.adaptive_avg_pool1d(x, self.adaptive_pool_size)
        return self.mlp(x.flatten(start_dim=1))


class UCountHead(nn.Module):
    def __init__(
        self,
        input_channels: int,
        *,
        conv_hidden_dims: list[int],
        mlp_hidden_dims: list[int],
        kernel_size: int,
        dropout: float,
        norm_type: str,
        activation: str,
        mlp_norm_type: str = "layer",
        num_groups: int = 8,
        global_pool: str = "avgmax",
    ):
        super().__init__()
        self.global_pool = global_pool.lower()
        if self.global_pool not in {"avg", "max", "avgmax"}:
            raise ValueError(f"Unsupported global_pool: {global_pool}")
        conv_layers: list[nn.Module] = []
        channels = input_channels
        for hidden in conv_hidden_dims:
            conv_layers.extend(
                [
                    nn.Conv1d(channels, hidden, kernel_size=kernel_size, padding="same", bias=False),
                    make_norm(norm_type, hidden, num_groups),
                    make_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            channels = hidden
        self.conv_stack = nn.Sequential(*conv_layers)

        mlp_layers: list[nn.Module] = []
        mlp_input_dim = channels * (2 if self.global_pool == "avgmax" else 1)
        for hidden in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(mlp_input_dim, hidden),
                    make_dense_norm(mlp_norm_type, hidden, num_groups),
                    make_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            mlp_input_dim = hidden
        mlp_layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def _pool(self, x: Tensor) -> Tensor:
        if self.global_pool == "avg":
            return x.mean(dim=-1)
        if self.global_pool == "max":
            return x.amax(dim=-1)
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self._pool(self.conv_stack(x)))
