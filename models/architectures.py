"""Neural architectures used in the AMAC manuscript."""

from __future__ import annotations
from typing import Mapping, Sequence
import torch


class CorrectnessMLP(torch.nn.Module):
    """Two-output MLP used by the frozen correctness checkpoints."""

    def __init__(self, input_dim: int, hidden: Sequence[int] = (64, 32), dropout: float = 0.1) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[0], hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[1], 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class MaskedFusionRegressor(torch.nn.Module):
    """Masked multimodal Transformer used in the nonlinear robustness study."""

    def __init__(self, dims: Sequence[int], config: Mapping[str, int | float]) -> None:
        super().__init__()
        width = int(config["token_dim"])
        dropout = float(config["dropout"])
        self.projections = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(dim, width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(width),
            )
            for dim in dims
        ])
        self.modality_embedding = torch.nn.Parameter(torch.empty(3, width))
        torch.nn.init.normal_(self.modality_embedding, std=0.02)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=width,
            nhead=int(config["attention_heads"]),
            dim_feedforward=int(config["feedforward_dim"]),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            layer, num_layers=int(config["transformer_layers"])
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(width + 3, int(config["fusion_hidden"])),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(int(config["fusion_hidden"]), 1),
        )

    def forward(self, modalities: Sequence[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [project(value) for project, value in zip(self.projections, modalities)],
            dim=1,
        ) + self.modality_embedding.unsqueeze(0)
        tokens = tokens * mask.unsqueeze(-1)
        encoded = self.transformer(tokens, src_key_padding_mask=~mask.bool())
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(torch.cat((pooled, mask), dim=1)).squeeze(1)
