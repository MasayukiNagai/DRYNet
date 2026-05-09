from __future__ import annotations

import torch
from torch import Tensor, nn


class ProCapNet(nn.Module):
    """BPNet-style ProCapNet architecture used for PRO-cap profile/count prediction."""

    def __init__(
        self,
        *,
        n_filters: int = 512,
        n_layers: int = 8,
        n_outputs: int = 2,
        trimming: int = (2114 - 1000) // 2,
    ) -> None:
        super().__init__()
        self.n_filters = int(n_filters)
        self.n_layers = int(n_layers)
        self.n_outputs = int(n_outputs)
        self.trimming = int(trimming)

        self.iconv = nn.Conv1d(4, self.n_filters, kernel_size=21, padding=10)
        self.rconvs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.n_filters,
                    self.n_filters,
                    kernel_size=3,
                    padding=2**i,
                    dilation=2**i,
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(self.n_layers + 1)])
        self.profile_kernel_size = 75
        self.fconv = nn.Conv1d(self.n_filters, self.n_outputs, kernel_size=self.profile_kernel_size)
        self.linear = nn.Linear(self.n_filters, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, 4, length), got {tuple(x.shape)}")
        if x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 sequence channels, got shape {tuple(x.shape)}")

        start = self.trimming
        end = x.shape[2] - self.trimming

        x = self.relus[0](self.iconv(x))
        for i, conv in enumerate(self.rconvs):
            x_conv = self.relus[i + 1](conv(x))
            x = x + x_conv

        profile_features = x[:, :, start - self.profile_kernel_size // 2 : end + self.profile_kernel_size // 2]
        profile_logits = self.fconv(profile_features)
        log_counts = self.linear(profile_features.mean(dim=2)).reshape(x.shape[0], 1)
        return profile_logits, log_counts
