from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .config import normalize_config
from .layers import (
    Bottleneck,
    EmbeddingProjector,
    IdentityProjector,
    ProfileHead,
    SequenceDecoder,
    SequenceEncoder,
    UCountHead,
    YCountHead,
    center_crop_1d,
)


class CAPY(nn.Module):
    """Count And Profile Y-net model."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = normalize_config(config)
        model_cfg = self.config["model"]

        self.input_length = model_cfg["input_length"]
        self.output_length = model_cfg["output_length"]
        self.profile_embedding_length = self.output_length + model_cfg["profile_head"]["kernel_size"] - 1

        self.encoder = SequenceEncoder(
            input_channels=model_cfg["input_channels"],
            dna_embedder_channels=model_cfg["dna_embedder_channels"],
            encoder_channels=model_cfg["encoder_channels"],
            pool_size=model_cfg["pool_size"],
            norm_type=model_cfg["norm_type"],
            num_groups=model_cfg["num_groups"],
            activation=model_cfg["activation"],
            dropout=model_cfg["dropout"],
        )
        self.bottleneck = Bottleneck(
            channels=model_cfg["encoder_channels"][-1],
            bottleneck_cfg=model_cfg["bottleneck"],
            fallback_norm_type=model_cfg["norm_type"],
            fallback_num_groups=model_cfg["num_groups"],
            fallback_activation=model_cfg["activation"],
        )
        decoder_channels = list(model_cfg["decoder_channels"])
        self.decoder = SequenceDecoder(
            dna_embedder_channels=model_cfg["dna_embedder_channels"],
            encoder_channels=model_cfg["encoder_channels"],
            decoder_channels=decoder_channels,
            norm_type=model_cfg["norm_type"],
            num_groups=model_cfg["num_groups"],
            activation=model_cfg["activation"],
            dropout=model_cfg["dropout"],
        )

        if model_cfg["embedding_projector"]["enabled"]:
            self.encoder_projector = EmbeddingProjector(
                model_cfg["encoder_channels"][-1],
                model_cfg["encoder_channels"][-1],
                norm_type=model_cfg["norm_type"],
                num_groups=model_cfg["num_groups"],
                activation=model_cfg["activation"],
                dropout=model_cfg["dropout"],
            )
            self.bottleneck_projector = EmbeddingProjector(
                model_cfg["encoder_channels"][-1],
                model_cfg["encoder_channels"][-1],
                norm_type=model_cfg["norm_type"],
                num_groups=model_cfg["num_groups"],
                activation=model_cfg["activation"],
                dropout=model_cfg["dropout"],
            )
            self.output_embedder = EmbeddingProjector(
                decoder_channels[-1],
                model_cfg["output_embedder_channels"],
                norm_type=model_cfg["norm_type"],
                num_groups=model_cfg["num_groups"],
                activation=model_cfg["activation"],
                dropout=model_cfg["dropout"],
            )
            embedding_dims = {
                "encoder": model_cfg["encoder_channels"][-1],
                "bottleneck": model_cfg["encoder_channels"][-1],
                "decoder": model_cfg["output_embedder_channels"],
            }
        else:
            self.encoder_projector = IdentityProjector()
            self.bottleneck_projector = IdentityProjector()
            self.output_embedder = IdentityProjector()
            embedding_dims = {
                "encoder": model_cfg["encoder_channels"][-1],
                "bottleneck": model_cfg["encoder_channels"][-1],
                "decoder": decoder_channels[-1],
            }

        self.profile_head_source = model_cfg["profile_head"]["source"]
        self.count_head_source = model_cfg["count_head"]["source"]
        self.profile_feature_dimension = embedding_dims[self.profile_head_source]
        self.count_feature_dimension = embedding_dims[self.count_head_source]

        self.profile_head = ProfileHead(
            input_channels=self.profile_feature_dimension,
            num_outputs=model_cfg["profile_head"]["num_outputs"],
            kernel_size=model_cfg["profile_head"]["kernel_size"],
            output_length=model_cfg["output_length"],
        )
        count_cfg = model_cfg["count_head"]
        if count_cfg["type"] == "ynet":
            self.count_head = YCountHead(
                input_channels=self.count_feature_dimension,
                conv_hidden_dims=count_cfg["conv_hidden_dims"],
                mlp_hidden_dims=count_cfg["mlp_hidden_dims"],
                kernel_size=count_cfg["kernel_size"],
                pool_size=count_cfg["pool_size"],
                adaptive_pool_size=count_cfg["adaptive_pool_size"],
                dropout=count_cfg["dropout"],
                norm_type=count_cfg["norm_type"],
                mlp_norm_type=count_cfg["mlp_norm_type"],
                num_groups=count_cfg["num_groups"],
                activation=model_cfg["activation"],
            )
        else:
            self.count_head = UCountHead(
                input_channels=self.count_feature_dimension,
                conv_hidden_dims=count_cfg["conv_hidden_dims"],
                mlp_hidden_dims=count_cfg["mlp_hidden_dims"],
                kernel_size=count_cfg["kernel_size"],
                dropout=count_cfg["dropout"],
                norm_type=count_cfg["norm_type"],
                mlp_norm_type=count_cfg["mlp_norm_type"],
                num_groups=count_cfg["num_groups"],
                activation=model_cfg["activation"],
                global_pool=count_cfg["global_pool"],
            )

    def _prepare_input(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (batch, length, channels) or (batch, channels, length), got {tuple(x.shape)}")
        if x.shape[1] == self.input_length:
            return x.transpose(1, 2)
        if x.shape[2] == self.input_length:
            return x
        raise ValueError(f"Input length must match configured input_length={self.input_length}. Got shape {tuple(x.shape)}")

    def encode_all(self, x: Tensor) -> dict[str, Tensor]:
        x = self._prepare_input(x)
        encoder_raw, intermediates = self.encoder(x)
        encoder_embeds = self.encoder_projector(encoder_raw)
        bottleneck_raw = self.bottleneck(encoder_raw)
        bottleneck_embeds = self.bottleneck_projector(bottleneck_raw)
        decoder_raw = self.decoder(bottleneck_raw, intermediates)
        decoder_embeds = self.output_embedder(decoder_raw)
        decoder_embeds = center_crop_1d(decoder_embeds, self.profile_embedding_length)
        return {
            "encoder_embeds": encoder_embeds,
            "bottleneck_embeds": bottleneck_embeds,
            "decoder_embeds": decoder_embeds,
        }

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.encode_all(x)
        profile_embeds = embeddings[f"{self.profile_head_source}_embeds"]
        count_embeds = embeddings[f"{self.count_head_source}_embeds"]
        return self.profile_head(profile_embeds), self.count_head(count_embeds)
