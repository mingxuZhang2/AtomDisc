"""Configuration objects for Stage-1 GNN+VQ components.

The defaults are intentionally minimal so that downstream scripts can
override them via CLI/JSON/YAML without editing the package code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stage1Config:
    """Hyperparameters required to construct the frozen GNN/VQ modules."""

    num_layer: int = 5
    emb_dim: int = 300
    JK: str = "last"
    dropout_ratio: float = 0.1
    gnn_type: str = "gin"
    codebook_size: int = 512

    def to_dict(self) -> dict:
        return {
            "num_layer": self.num_layer,
            "emb_dim": self.emb_dim,
            "JK": self.JK,
            "dropout_ratio": self.dropout_ratio,
            "gnn_type": self.gnn_type,
            "codebook_size": self.codebook_size,
        }


DEFAULT_STAGE1_CONFIG = Stage1Config()