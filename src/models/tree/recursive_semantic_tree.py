from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class RecursiveSemanticTree(nn.Module):
    """Shared recursive semantic tree with modality-specific writeback layers."""

    def __init__(
        self,
        d_z: int,
        tree_prototypes: list[int] | tuple[int, ...],
        assignment_temperature: float = 1.0,
        upward_alpha: float = 0.5,
        writeback_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        if d_z <= 0:
            raise ValueError("d_z must be positive")
        if not tree_prototypes:
            raise ValueError("tree_prototypes must be non-empty")
        if any(int(count) <= 0 for count in tree_prototypes):
            raise ValueError("all tree_prototypes entries must be positive")
        if assignment_temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive")
        if not (0.0 <= upward_alpha <= 1.0):
            raise ValueError("upward_alpha must be in [0, 1]")
        if not (0.0 <= writeback_gamma <= 1.0):
            raise ValueError("writeback_gamma must be in [0, 1]")
        self.d_z = int(d_z)
        self.tree_prototypes = [int(count) for count in tree_prototypes]
        self.tree_levels = len(self.tree_prototypes)
        self.assignment_temperature = float(assignment_temperature)
        self.upward_alpha = float(upward_alpha)
        self.writeback_gamma = float(writeback_gamma)

        self.prototypes = nn.ParameterList(
            [nn.Parameter(torch.empty(count, self.d_z)) for count in self.tree_prototypes]
        )
        self.transition_logits = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.tree_prototypes[level], self.tree_prototypes[level - 1]))
                for level in range(1, self.tree_levels)
            ]
        )
        self.image_writeback = nn.ModuleList([nn.Linear(self.d_z, self.d_z) for _ in range(self.tree_levels)])
        self.text_writeback = nn.ModuleList([nn.Linear(self.d_z, self.d_z) for _ in range(self.tree_levels)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for prototype in self.prototypes:
            nn.init.normal_(prototype, mean=0.0, std=0.05)
        for transition in self.transition_logits:
            nn.init.normal_(transition, mean=0.0, std=0.02)
        for layer in list(self.image_writeback) + list(self.text_writeback):
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, Z_I: torch.Tensor, Z_T: torch.Tensor) -> dict[str, Any]:
        _check_pair_inputs(Z_I, Z_T, self.d_z)
        assignments_I = [_stable_assignment(Z_I, prototype, self.assignment_temperature) for prototype in self.prototypes]
        assignments_T = [_stable_assignment(Z_T, prototype, self.assignment_temperature) for prototype in self.prototypes]

        Ubar: list[torch.Tensor] = []
        for level in range(self.tree_levels):
            U_I = _weighted_nodes(assignments_I[level], Z_I, f"Pi_I[{level}]")
            U_T = _weighted_nodes(assignments_T[level], Z_T, f"Pi_T[{level}]")
            shared = 0.5 * (U_I + U_T)
            _check_2d_finite(shared, f"Ubar[{level}]")
            Ubar.append(shared)

        U: list[torch.Tensor] = [Ubar[0]]
        transition_row_errors: list[float] = []
        for level in range(1, self.tree_levels):
            transition = torch.softmax(self.transition_logits[level - 1], dim=1)
            row_error = torch.max(torch.abs(transition.sum(dim=1) - 1.0)).detach().item()
            transition_row_errors.append(float(row_error))
            parent_from_child = transition @ U[level - 1]
            current = self.upward_alpha * Ubar[level] + (1.0 - self.upward_alpha) * parent_from_child
            _check_2d_finite(current, f"U[{level}]")
            U.append(current)

        Y_I = self._writeback(assignments_I, U, self.image_writeback, "Y_I")
        Y_T = self._writeback(assignments_T, U, self.text_writeback, "Y_T")
        diagnostics = self._diagnostics(assignments_I, assignments_T, Ubar, U, transition_row_errors, Z_I, Z_T, Y_I, Y_T)
        return {
            "Y_I": Y_I,
            "Y_T": Y_T,
            "assignments_I": assignments_I,
            "assignments_T": assignments_T,
            "Ubar": Ubar,
            "U": U,
            "diagnostics": diagnostics,
        }

    def _writeback(
        self,
        assignments: list[torch.Tensor],
        U: list[torch.Tensor],
        layers: nn.ModuleList,
        name: str,
    ) -> torch.Tensor:
        current: torch.Tensor | None = None
        for level in reversed(range(self.tree_levels)):
            base = assignments[level] @ U[level]
            value = layers[level](base)
            current = value if current is None else value + self.writeback_gamma * current
            _check_2d_finite(current, f"{name}[{level}]")
        if current is None:
            raise RuntimeError(f"{name} writeback did not run")
        return current

    def _diagnostics(
        self,
        assignments_I: list[torch.Tensor],
        assignments_T: list[torch.Tensor],
        Ubar: list[torch.Tensor],
        U: list[torch.Tensor],
        transition_row_errors: list[float],
        Z_I: torch.Tensor,
        Z_T: torch.Tensor,
        Y_I: torch.Tensor,
        Y_T: torch.Tensor,
    ) -> dict[str, Any]:
        row_errors_I = [_row_sum_error(assignment) for assignment in assignments_I]
        row_errors_T = [_row_sum_error(assignment) for assignment in assignments_T]
        usage_I = [_prototype_usage(assignment) for assignment in assignments_I]
        usage_T = [_prototype_usage(assignment) for assignment in assignments_T]
        combined_usage = [0.5 * (usage_I[level] + usage_T[level]) for level in range(self.tree_levels)]
        entropy_I = [_assignment_entropy(assignment) for assignment in assignments_I]
        entropy_T = [_assignment_entropy(assignment) for assignment in assignments_T]
        usage_threshold = 1.0
        empty_threshold = 1e-6
        return {
            "tree_level_count": self.tree_levels,
            "prototype_shapes": [list(prototype.shape) for prototype in self.prototypes],
            "shared_ubar_shapes": [list(node.shape) for node in Ubar],
            "shared_u_shapes": [list(node.shape) for node in U],
            "assignment_entropy_I": entropy_I,
            "assignment_entropy_T": entropy_T,
            "assignment_entropy": [
                0.5 * (entropy_I[level] + entropy_T[level]) for level in range(self.tree_levels)
            ],
            "effective_prototypes_used_I": [
                int(torch.sum(usage > usage_threshold).detach().item()) for usage in usage_I
            ],
            "effective_prototypes_used_T": [
                int(torch.sum(usage > usage_threshold).detach().item()) for usage in usage_T
            ],
            "effective_prototypes_used": [
                int(torch.sum(usage > usage_threshold).detach().item()) for usage in combined_usage
            ],
            "empty_prototype_count_I": [
                int(torch.sum(usage <= empty_threshold).detach().item()) for usage in usage_I
            ],
            "empty_prototype_count_T": [
                int(torch.sum(usage <= empty_threshold).detach().item()) for usage in usage_T
            ],
            "empty_prototype_count": [
                int(torch.sum(usage <= empty_threshold).detach().item()) for usage in combined_usage
            ],
            "prototype_usage_I": [_usage_summary(usage) for usage in usage_I],
            "prototype_usage_T": [_usage_summary(usage) for usage in usage_T],
            "prototype_usage": [_usage_summary(usage) for usage in combined_usage],
            "prototype_effective_usage_threshold": usage_threshold,
            "prototype_empty_usage_threshold": empty_threshold,
            "y_z_norm_ratio_I": _norm_ratio(Y_I, Z_I, "Y_I", "Z_I"),
            "y_z_norm_ratio_T": _norm_ratio(Y_T, Z_T, "Y_T", "Z_T"),
            "y_z_norm_ratio": 0.5 * (_norm_ratio(Y_I, Z_I, "Y_I", "Z_I") + _norm_ratio(Y_T, Z_T, "Y_T", "Z_T")),
            "assignment_row_sum_max_error_I": max(row_errors_I),
            "assignment_row_sum_max_error_T": max(row_errors_T),
            "assignment_row_sum_max_error": max(max(row_errors_I), max(row_errors_T)),
            "assignment_row_sum_error_I": row_errors_I,
            "assignment_row_sum_error_T": row_errors_T,
            "transition_row_sum_max_error": max(transition_row_errors) if transition_row_errors else 0.0,
            "writeback_modality_specific": True,
            "shared_semantic_nodes": "Ubar_and_U",
        }


def _stable_assignment(Z: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (Z @ prototypes.transpose(0, 1)) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values
    assignment = F.softmax(logits, dim=1)
    _check_2d_finite(assignment, "assignment")
    return assignment


def _weighted_nodes(assignment: torch.Tensor, Z: torch.Tensor, name: str) -> torch.Tensor:
    counts = assignment.sum(dim=0)
    if not torch.isfinite(counts).all():
        raise RuntimeError(f"{name} assignment counts contain NaN or Inf")
    if torch.any(counts <= 0.0):
        raise RuntimeError(f"{name} assignment has an empty prototype")
    return assignment.transpose(0, 1) @ Z / counts.unsqueeze(1)


def _row_sum_error(assignment: torch.Tensor) -> float:
    return float(torch.max(torch.abs(assignment.sum(dim=1) - 1.0)).detach().item())


def _assignment_entropy(assignment: torch.Tensor) -> float:
    entropy = -(assignment * torch.log(assignment.clamp_min(1e-12))).sum(dim=1)
    return float(entropy.mean().detach().item())


def _prototype_usage(assignment: torch.Tensor) -> torch.Tensor:
    usage = assignment.sum(dim=0)
    if not torch.isfinite(usage).all():
        raise RuntimeError("prototype usage contains NaN or Inf")
    return usage


def _usage_summary(usage: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(usage.min().detach().item()),
        "max": float(usage.max().detach().item()),
        "mean": float(usage.mean().detach().item()),
    }


def _norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor, numerator_name: str, denominator_name: str) -> float:
    num_norm = torch.linalg.vector_norm(numerator)
    den_norm = torch.linalg.vector_norm(denominator)
    if not torch.isfinite(num_norm) or not torch.isfinite(den_norm):
        raise RuntimeError(f"{numerator_name}/{denominator_name} norm contains NaN or Inf")
    if den_norm <= 0.0:
        raise RuntimeError(f"{denominator_name} norm must be positive")
    return float((num_norm / den_norm).detach().item())


def _check_pair_inputs(Z_I: torch.Tensor, Z_T: torch.Tensor, d_z: int) -> None:
    _check_2d_finite(Z_I, "Z_I")
    _check_2d_finite(Z_T, "Z_T")
    if Z_I.shape != Z_T.shape:
        raise RuntimeError(f"Z_I and Z_T shapes must match, got {tuple(Z_I.shape)} and {tuple(Z_T.shape)}")
    if Z_I.shape[1] != d_z:
        raise RuntimeError(f"expected d_z={d_z}, got {Z_I.shape[1]}")


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
