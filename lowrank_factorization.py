import torch
import numpy as np
from typing import Dict, Tuple, List


def compute_optimal_rank(singular_values: torch.Tensor, energy_threshold: float = 0.95) -> int:
    squared_sv = singular_values ** 2
    cumulative = torch.cumsum(squared_sv, dim=0)
    total = torch.sum(squared_sv)
    normalized = cumulative / total

    r = torch.searchsorted(normalized, energy_threshold).item() + 1
    return min(r, len(singular_values))


def low_rank_factorize(weight: torch.Tensor, energy_threshold: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    original_shape = weight.shape

    if weight.dim() > 2:
        weight = weight.reshape(-1, weight.shape[-1])

    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

    r = compute_optimal_rank(S, energy_threshold)

    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]

    U_r = U_r.reshape(original_shape[0], r)
    Vt_r = Vt_r.reshape(r, original_shape[-1])

    return U_r, S_r, Vt_r, r


def factorize_model_weights(weights: Dict[int, torch.Tensor], energy_threshold: float = 0.95) -> Dict:
    factors = {}

    for layer_idx, W in weights.items():
        U, S, Vt, r = low_rank_factorize(W, energy_threshold)
        factors[layer_idx] = {
            'U': U,
            'S': S,
            'Vt': Vt,
            'rank': r,
            'original_shape': W.shape,
            'energy_captured': (S ** 2).sum().item() / ((W ** 2).sum().item() + 1e-8)
        }

    return factors


def reconstruct_weight(factors: Dict) -> torch.Tensor:
    U = factors['U']
    S = factors['S']
    Vt = factors['Vt']

    return torch.matmul(U * S.unsqueeze(0), Vt)


def compute_compression_ratio(original_size: int, factors: Dict) -> float:
    U_size = sum(f['U'].numel() for f in factors.values())
    S_size = sum(f['S'].numel() for f in factors.values())
    Vt_size = sum(f['Vt'].numel() for f in factors.values())

    low_rank_size = U_size + S_size + Vt_size
    return original_size / low_rank_size


if __name__ == "__main__":
    W = torch.randn(4096, 4096)

    U, S, Vt, r = low_rank_factorize(W, energy_threshold=0.95)
    print(f"Original shape: {W.shape}")
    print(f"Rank: {r}")
    print(f"U shape: {U.shape}, S shape: {S.shape}, Vt shape: {Vt.shape}")

    reconstructed = torch.matmul(U * S.unsqueeze(0), Vt)
    error = torch.norm(W - reconstructed) / torch.norm(W)
    print(f"Reconstruction error: {error:.6f}")

    weights = {0: torch.randn(4096, 4096), 1: torch.randn(4096, 4096)}
    factors = factorize_model_weights(weights)
    for layer_idx, f in factors.items():
        print(f"Layer {layer_idx}: rank={f['rank']}, energy={f['energy_captured']:.4f}")