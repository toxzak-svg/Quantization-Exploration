import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
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

        self.bit_widths = nn.Parameter(torch.full((num_tensors,), initial_bits))

    def forward(self) -> List[int]:
        return torch.round(self.bit_widths).clamp(0, 2).int().tolist()

    def regularization_loss(self) -> torch.Tensor:
        return self.bit_penalty * torch.sum(torch.abs(self.bit_widths))


class QuantizedTensor:
    def __init__(self, data: torch.Tensor, config: TensorQuantConfig):
        self.data = data
        self.config = config


class AdaQATrainer:
    def __init__(
        self,
        factors: Dict,
        perplexity_fn,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.factors = factors
        self.perplexity_fn = perplexity_fn

        self.num_tensors = sum(len(f) for f in factors.values())
        self.bit_search = AdaQATBitWidthSearch(self.num_tensors).to(device)

        self.optimizer = torch.optim.AdamW(self.bit_search.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def allocate_bits(self, bit_width: int) -> Tuple[int, int, int]:
        if bit_width == 0:
            return 0, 0, 0
        elif bit_width == 1:
            return 1, 0, 0
        else:
            return 1, 1, 0

    def train_step(self, model_weights: Dict) -> Tuple[float, float, List[int]]:
        self.bit_search.train()

        bit_allocations = self.bit_search()
        bit_penalty = self.bit_search.regularization_loss()

        ppl = self.perplexity_fn(self.factors, bit_allocations)

        total_bits = sum(bit_allocations) / len(bit_allocations)

        loss = ppl + bit_penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return ppl.item(), bit_penalty.item(), bit_allocations

    def train(self, model_weights: Dict, num_epochs: int = 2):
        best_ppl = float('inf')
        best_allocation = None

        for epoch in range(num_epochs):
            ppl, bit_penalty, allocations = self.train_step(model_weights)
            avg_bits = sum(allocations) / len(allocations)

            if ppl < best_ppl:
                best_ppl = ppl
                best_allocation = allocations.copy()

            print(f"Epoch {epoch+1}/{num_epochs} - PPL: {ppl:.4f}, Bit penalty: {bit_penalty:.6f}, Avg bits: {avg_bits:.2f}")

        return best_allocation, best_ppl

    def get_final_allocation(self) -> List[Tuple[int, int, int]]:
        final_bits = self.bit_search()
        return [self.allocate_bits(b) for b in final_bits]


def apply_quantization(factors: Dict, allocation: List[int], uv_bits: int = 1, sigma_bits: int = 2) -> Dict:
    quantized = {}

    for layer_idx, layer_factors in factors.items():
        bits = allocation[layer_idx] if layer_idx < len(allocation) else 0

        q_U, _, _ = quantize_to_ternary(layer_factors['U'])
        q_V, _, _ = quantize_to_ternary(layer_factors['Vt'])
        q_S, _, _ = quantize_sigma(layer_factors['S'], sigma_bits)

        quantized[layer_idx] = {
            'U': q_U,
            'V': q_V,
            'S': q_S,
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
    def dummy_perplexity_fn(factors, allocations):
        return torch.tensor(10.5 + torch.rand(1).item() * 0.5)

    factors = {
        0: {
            'U': torch.randn(4096, 16),
            'S': torch.randn(16),
            'Vt': torch.randn(16, 4096),
            'original_shape': (4096, 4096)
        }
    }

    trainer = AdaQATrainer(factors, dummy_perplexity_fn)
    best_allocation, best_ppl = trainer.train({}, num_epochs=10)

    print(f"Best allocation: {best_allocation}")
    print(f"Best PPL: {best_ppl:.4f}")