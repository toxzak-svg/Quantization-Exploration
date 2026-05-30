import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math


@dataclass
class TensorQuantConfig:
    num_bits: int
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None


class AdaQATBitWidthSearch(nn.Module):
    def __init__(self, num_tensors: int, initial_bits: float = 1.0, bit_penalty: float = 1e-4):
        super().__init__()
        self.num_tensors = num_tensors
        self.bit_penalty = bit_penalty
        self.bit_logits = nn.Parameter(torch.full((num_tensors, 3), initial_bits / 2))
        self.register_buffer('bit_values', torch.tensor([0.0, 1.0, 2.0]))

    def soft_allocation(self) -> torch.Tensor:
        probs = F.softmax(self.bit_logits, dim=-1)
        return (probs * self.bit_values).sum(dim=-1)

    def hard_allocation(self) -> torch.Tensor:
        with torch.no_grad():
            return self.soft_allocation().round().clamp(0, 2).int().tolist()

    def forward(self) -> torch.Tensor:
        soft = self.soft_allocation()
        hard = soft.round().clamp(0, 2).detach()
        return hard + (soft - soft.detach())

    def regularization_loss(self) -> torch.Tensor:
        return self.bit_penalty * self.soft_allocation().sum()


class AdaQATrainer:
    def __init__(
        self,
        factors: Dict,
        loss_fn: Callable,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.factors = factors
        self.loss_fn = loss_fn
        self.num_tensors = sum(len(f) for f in factors.values())
        self.bit_search = AdaQATBitWidthSearch(self.num_tensors).to(device)
        self.optimizer = torch.optim.AdamW(self.bit_search.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def train_step(self, model_weights: Dict) -> Tuple[float, float, List[int]]:
        self.bit_search.train()
        alloc = self.bit_search()
        penalty = self.bit_search.regularization_loss()
        surrogate_loss = self.loss_fn(self.factors, alloc)
        loss = surrogate_loss + penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        hard = self.bit_search.hard_allocation()
        return surrogate_loss.item(), penalty.item(), hard

    def train(self, model_weights: Dict, num_epochs: int = 2):
        best_loss = float('inf')
        best_allocation = None
        for epoch in range(num_epochs):
            loss, penalty, allocations = self.train_step(model_weights)
            avg_bits = sum(allocations) / len(allocations) if allocations else 0
            if loss < best_loss:
                best_loss = loss
                best_allocation = allocations.copy()
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}, Penalty: {penalty:.6f}, Avg bits: {avg_bits:.2f}")
        return best_allocation, best_loss


def reconstruction_loss(factors: Dict, allocations: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    count = 0
    for layer_idx, layer_factors in factors.items():
        W = layer_factors.get('original', None)
        if W is None:
            continue
        U, S, Vt = layer_factors['U'], layer_factors['S'], layer_factors['Vt']
        bits = allocations[count].clamp(1, 2)
        qmax = 2 ** int(bits.round().item()) - 1
        scale = S.abs().max() / qmax if S.abs().max() > 0 else 1.0
        S_q = (S / scale).round().clamp(-qmax, qmax)
        S_recon = S_q * scale
        W_recon = U.float() @ torch.diag(S_recon) @ Vt.float()
        loss = loss + F.mse_loss(W_recon, W.float())
        count += 1
    return loss / max(count, 1)


def apply_quantization(factors: Dict, allocation: List[int], uv_bits: int = 1, sigma_bits: int = 2) -> Dict:
    quantized = {}
    for layer_idx, layer_factors in factors.items():
        bits = allocation[layer_idx] if layer_idx < len(allocation) else 0
        q_U, _, _ = quantize_to_ternary(layer_factors['U'])
        q_V, _, _ = quantize_to_ternary(layer_factors['Vt'])
        q_S, _, _ = quantize_sigma(layer_factors['S'], sigma_bits)
        quantized[layer_idx] = {
            'U': q_U, 'V': q_V, 'S': q_S,
            'bits': bits,
            'original_shape': layer_factors['original_shape']
        }
    return quantized


def quantize_to_ternary(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = x.abs().max()
    normalized = x / scale
    ternary = torch.sign(normalized)
    ternary[ternary == 0] = 1
    return ternary.to(torch.int8), scale, torch.tensor(0, dtype=torch.int8)


def quantize_sigma(sigma: torch.Tensor, num_bits: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_val = sigma.abs().max()
    scale = max_val / (2 ** (num_bits - 1) - 1)
    normalized = (sigma / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
    return normalized.to(torch.int8), scale, max_val


if __name__ == "__main__":
    factors = {
        0: {
            'original': torch.randn(4096, 4096),
            'U': torch.randn(4096, 16),
            'S': torch.randn(16),
            'Vt': torch.randn(16, 4096),
            'original_shape': (4096, 4096)
        }
    }
    trainer = AdaQATrainer(factors, reconstruction_loss)
    best_allocation, best_loss = trainer.train({}, num_epochs=10)
    print(f"Best allocation: {best_allocation}")
    print(f"Best loss: {best_loss:.4f}")
