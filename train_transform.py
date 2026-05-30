import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LearnableTransform(nn.Module):
    def __init__(self, dim: int, orthogonality_reg: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.orthogonality_reg = orthogonality_reg

        self.scale = nn.Parameter(torch.ones(dim))
        self.rotation = nn.Parameter(torch.eye(dim))
        nn.init.orthogonal_(self.rotation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x * self.scale
        return torch.matmul(scaled, self.rotation)

    def orthogonality_loss(self) -> torch.Tensor:
        I = torch.eye(self.dim, device=self.rotation.device)
        return F.mse_loss(torch.matmul(self.rotation.T, self.rotation), I)


class BinaryCodebook(nn.Module):
    def __init__(self, codebook_size: int = 256, dim: int = 128):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distances = torch.cdist(x, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)
        return quantized, indices


class TransformLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, codebook_dim: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codebook_dim = codebook_dim

        self.transform = LearnableTransform(in_features)
        self.codebook = BinaryCodebook(codebook_size=256, dim=codebook_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed = self.transform(x)

        if transformed.shape[-1] != self.codebook_dim:
            proj = nn.Linear(transformed.shape[-1], self.codebook_dim, device=transformed.device)
            transformed = proj(transformed)

        quantized, indices = self.codebook(transformed)
        return transformed, quantized, indices


class BTCLLMTrainer:
    def __init__(
        self,
        layer_sizes: List[Tuple[int, int]],
        codebook_dim: int = 128,
        lr: float = 1e-3,
        orthogonality_reg: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.layer_sizes = layer_sizes
        self.codebook_dim = codebook_dim
        self.orthogonality_reg = orthogonality_reg

        self.layers = nn.ModuleList([
            TransformLayer(in_f, out_f, codebook_dim) for in_f, out_f in layer_sizes
        ]).to(device)

        self.codebook_optimizer = torch.optim.Adam(
            self.layers.parameters(), lr=lr
        )

    def train_epoch(self, weight_data: Dict[int, torch.Tensor], batch_size: int = 256) -> float:
        total_loss = 0.0
        num_batches = 0

        for layer_idx, (in_f, out_f) in enumerate(self.layer_sizes):
            if layer_idx not in weight_data:
                continue

            W = weight_data[layer_idx].to(self.device)
            flat_W = W.view(-1, in_f)

            indices = torch.randperm(flat_W.shape[0], device=self.device)
            num_batches_layer = 0

            for start in range(0, flat_W.shape[0], batch_size):
                end = min(start + batch_size, flat_W.shape[0])
                batch_idx = indices[start:end]
                batch = flat_W[batch_idx]

                self.codebook_optimizer.zero_grad()

                transformed, quantized, _ = self.layers[layer_idx](batch)

                recon_loss = F.mse_loss(quantized, transformed)
                ortho_loss = self.layers[layer_idx].transform.orthogonality_loss()

                loss = recon_loss + self.orthogonality_reg * ortho_loss
                loss.backward()

                self.codebook_optimizer.step()

                total_loss += recon_loss.item()
                num_batches += 1
                num_batches_layer += 1

        return total_loss / max(num_batches, 1)

    def get_transformed_weights(self, weight_data: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        transformed = {}
        for layer_idx, (in_f, out_f) in enumerate(self.layer_sizes):
            if layer_idx not in weight_data:
                continue

            W = weight_data[layer_idx].to(self.device)
            with torch.no_grad():
                transformed[layer_idx] = self.layers[layer_idx].transform(W).cpu()
        return transformed

    def save(self, path: str):
        torch.save({
            'layers': self.layers.state_dict(),
            'layer_sizes': self.layer_sizes,
            'codebook_dim': self.codebook_dim
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        checkpoint = torch.load(path, map_location=device)
        trainer = cls(
            layer_sizes=checkpoint['layer_sizes'],
            codebook_dim=checkpoint['codebook_dim'],
            device=device
        )
        trainer.layers.load_state_dict(checkpoint['layers'])
        return trainer


class QuantizationFunctions:
    @staticmethod
    def quantize_ternary(x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if scale is None:
            scale = x.abs().max()
        normalized = x / scale
        ternary = torch.sign(normalized)
        ternary[ternary == 0] = 1
        return ternary.to(torch.int8), scale, torch.tensor(0, dtype=torch.int8)

    @staticmethod
    def quantize_sigma(sigma: torch.Tensor, num_bits: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_val = sigma.abs().max()
        scale = max_val / (2 ** (num_bits - 1) - 1)
        normalized = (sigma / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
        return normalized.to(torch.int8), scale, max_val

    @staticmethod
    def dequantize_ternary(q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return q.float() * scale


if __name__ == "__main__":
    layer_sizes = [(4096, 4096), (4096, 4096)]
    trainer = BTCLLMTrainer(layer_sizes, codebook_dim=128)

    dummy_weights = {
        0: torch.randn(4096, 4096),
        1: torch.randn(4096, 4096)
    }

    for epoch in range(5):
        loss = trainer.train_epoch(dummy_weights)
        print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

    trainer.save("checkpoints/btc_llm_transforms.pt")
    print("Training complete. Checkpoints saved.")