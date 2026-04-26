from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.wrappers.cross_modal_hash_net import CrossModalHashNet


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4A-1 smoke requires cuda:0; CPU fallback is not allowed")
    device = torch.device("cuda:0")
    torch.manual_seed(20260426)
    torch.cuda.manual_seed_all(20260426)

    N = 64
    input_dim = 512
    d_z = 64
    bit = 16
    tree_prototypes = [16, 8]
    graph_k = 5

    X_I = F.normalize(torch.randn(N, input_dim, device=device, dtype=torch.float32), p=2, dim=1)
    X_T = F.normalize(torch.randn(N, input_dim, device=device, dtype=torch.float32), p=2, dim=1)
    model = CrossModalHashNet(
        input_dim=input_dim,
        d_z=d_z,
        bit=bit,
        cheby_order=4,
        tree_prototypes=tree_prototypes,
        graph_k=graph_k,
        beta_tree_injection=1.0,
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model(X_I, X_T, bit=bit)

    _assert_shape(output["Z_I"], (N, d_z), "Z_I")
    _assert_shape(output["Z_T"], (N, d_z), "Z_T")
    _assert_shape(output["Y_I"], (N, d_z), "Y_I")
    _assert_shape(output["Y_T"], (N, d_z), "Y_T")
    _assert_shape(output["F_I"], (N, d_z), "F_I")
    _assert_shape(output["F_T"], (N, d_z), "F_T")
    _assert_shape(output["H_I"], (N, bit), "H_I")
    _assert_shape(output["H_T"], (N, bit), "H_T")
    if output["B_I"].shape != (N, bit) or output["B_T"].shape != (N, bit):
        raise RuntimeError("B_I/B_T shapes are invalid")
    if output["B_I"].dtype != torch.int8 or output["B_T"].dtype != torch.int8:
        raise RuntimeError("B_I/B_T must use torch.int8")

    b_unique = sorted({int(value) for value in torch.cat([output["B_I"].flatten(), output["B_T"].flatten()]).cpu().tolist()})
    if not set(b_unique).issubset({-1, 1}):
        raise RuntimeError(f"B has invalid values: {b_unique}")
    sign_rule_ok = bool(
        torch.equal(output["B_I"].to(torch.int16), torch.where(output["H_I"] >= 0.0, 1, -1).to(torch.int16))
        and torch.equal(output["B_T"].to(torch.int16), torch.where(output["H_T"] >= 0.0, 1, -1).to(torch.int16))
    )
    if not sign_rule_ok:
        raise RuntimeError("sign rule check failed")

    tree_diag = output["tree_diagnostics"]
    graph_diag = output["graph_diagnostics"]
    tree_error = float(tree_diag["assignment_row_sum_max_error"])
    degree_min = min(float(graph_diag["image"]["degree_min"]), float(graph_diag["text"]["degree_min"]))
    degree_max = max(float(graph_diag["image"]["degree_max"]), float(graph_diag["text"]["degree_max"]))
    if tree_error > 1e-5:
        raise RuntimeError(f"tree assignment row-sum error too high: {tree_error}")
    if degree_min <= 0.0:
        raise RuntimeError(f"graph degree min must be positive, got {degree_min}")

    print(f"N={N}")
    print(f"input_dim={input_dim}")
    print(f"d_z={d_z}")
    print(f"bit={bit}")
    print("tree_levels=2")
    print(f"tree_prototypes={tree_prototypes}")
    print(f"graph_k={graph_k}")
    print(f"H_I_shape={list(output['H_I'].shape)}")
    print(f"H_T_shape={list(output['H_T'].shape)}")
    print(f"B_I_shape={list(output['B_I'].shape)}")
    print(f"B_T_shape={list(output['B_T'].shape)}")
    print(f"B_unique_values={b_unique}")
    print(f"tree_assignment_row_sum_max_error={tree_error:.8g}")
    print(f"graph_degree_min={degree_min:.8g}")
    print(f"graph_degree_max={degree_max:.8g}")
    print("nan_inf_check=passed")
    print(f"sign_rule_check={str(sign_rule_ok).lower()}")
    print("smoke_result=passed")
    return 0


def _assert_shape(tensor: torch.Tensor, expected: tuple[int, int], name: str) -> None:
    if tensor.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


if __name__ == "__main__":
    raise SystemExit(main())
