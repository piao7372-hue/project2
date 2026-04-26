from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.encoders.chebykan import ChebyKAN
from src.models.graph.graph_refiner import GraphRefiner
from src.models.heads.hash_head import HashHead
from src.models.tree.recursive_semantic_tree import RecursiveSemanticTree


class CrossModalHashNet(nn.Module):
    """Stage 4 tree-graph hash mainline for synthetic smoke and later formal wiring."""

    def __init__(
        self,
        input_dim: int = 512,
        d_z: int = 256,
        bit: int = 16,
        cheby_order: int = 4,
        tree_prototypes: list[int] | tuple[int, ...] = (256, 64),
        graph_k: int = 15,
        beta_tree_injection: float = 1.0,
    ) -> None:
        super().__init__()
        if bit <= 0:
            raise ValueError("bit must be positive")
        self.input_dim = int(input_dim)
        self.d_z = int(d_z)
        self.bit = int(bit)
        self.graph_k = int(graph_k)
        self.tree_prototypes = [int(count) for count in tree_prototypes]

        self.image_chebykan = ChebyKAN(input_dim=self.input_dim, output_dim=self.d_z, order=cheby_order)
        self.text_chebykan = ChebyKAN(input_dim=self.input_dim, output_dim=self.d_z, order=cheby_order)
        self.semantic_tree = RecursiveSemanticTree(d_z=self.d_z, tree_prototypes=self.tree_prototypes)
        self.image_graph_refiner = GraphRefiner(
            d_z=self.d_z,
            graph_k=self.graph_k,
            beta_tree_injection=beta_tree_injection,
        )
        self.text_graph_refiner = GraphRefiner(
            d_z=self.d_z,
            graph_k=self.graph_k,
            beta_tree_injection=beta_tree_injection,
        )
        self.image_hash_head = HashHead(input_dim=self.d_z, bit=self.bit)
        self.text_hash_head = HashHead(input_dim=self.d_z, bit=self.bit)

    def forward(self, X_I: torch.Tensor, X_T: torch.Tensor, bit: int | None = None) -> dict[str, Any]:
        if bit is not None and int(bit) != self.bit:
            raise RuntimeError(f"CrossModalHashNet was initialized for bit={self.bit}, got bit={bit}")
        _check_inputs(X_I, X_T, self.input_dim)

        Z_I = self.image_chebykan(X_I)
        Z_T = self.text_chebykan(X_T)
        tree_output = self.semantic_tree(Z_I, Z_T)
        Y_I = tree_output["Y_I"]
        Y_T = tree_output["Y_T"]

        graph_I = self.image_graph_refiner(Z_I, Y_I)
        graph_T = self.text_graph_refiner(Z_T, Y_T)
        hash_I = self.image_hash_head(graph_I["graph_hidden"])
        hash_T = self.text_hash_head(graph_T["graph_hidden"])

        H_I = hash_I["H"]
        H_T = hash_T["H"]
        B_I = hash_I["B"]
        B_T = hash_T["B"]
        if torch.equal(H_I, H_T):
            raise RuntimeError("H_I and H_T are unexpectedly identical")
        if torch.equal(B_I, B_T):
            raise RuntimeError("B_I and B_T are unexpectedly identical")

        return {
            "Z_I": Z_I,
            "Z_T": Z_T,
            "Y_I": Y_I,
            "Y_T": Y_T,
            "F_I": graph_I["F"],
            "F_T": graph_T["F"],
            "H_I": H_I,
            "H_T": H_T,
            "B_I": B_I,
            "B_T": B_T,
            "tree_diagnostics": tree_output["diagnostics"],
            "graph_diagnostics": {
                "image": graph_I["diagnostics"],
                "text": graph_T["diagnostics"],
            },
        }


def _check_inputs(X_I: torch.Tensor, X_T: torch.Tensor, input_dim: int) -> None:
    _check_2d_finite(X_I, "X_I")
    _check_2d_finite(X_T, "X_T")
    if X_I.shape != X_T.shape:
        raise RuntimeError(f"X_I and X_T shapes must match, got {tuple(X_I.shape)} and {tuple(X_T.shape)}")
    if X_I.shape[1] != input_dim:
        raise RuntimeError(f"expected input_dim={input_dim}, got {X_I.shape[1]}")


def _check_2d_finite(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise RuntimeError(f"{name} must be non-empty")
    if not torch.is_floating_point(tensor):
        raise RuntimeError(f"{name} must be a floating point tensor")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
