import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Iterator
import json
import struct
from pathlib import Path
import numpy as np
import os


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


class TransformLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, codebook_dim: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codebook_dim = codebook_dim
        self.transform = LearnableTransform(in_features)
        self.codebook = nn.Parameter(torch.randn(codebook_size=256, dim=codebook_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = self.transform(x)
        distances = torch.cdist(transformed, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)
        return quantized, indices


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

        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)

    def train_epoch(self, weight_iter, num_batches_per_layer: int = 100) -> float:
        total_loss = 0.0
        num_batches = 0

        for layer_idx, (in_f, out_f) in enumerate(self.layer_sizes):
            W = next(weight_iter).float()

            flat_W = W.view(-1, in_f)
            batch_size = min(256, flat_W.shape[0])
            num_batches_layer = 0

            for _ in range(num_batches_per_layer):
                indices = torch.randperm(flat_W.shape[0], device=self.device)[:batch_size]
                batch = flat_W[indices].to(self.device)

                self.optimizer.zero_grad()

                transformed = self.layers[layer_idx].transform(batch)
                codebook = self.layers[layer_idx].codebook
                distances = torch.cdist(transformed, codebook)
                indices = distances.argmin(dim=-1)
                quantized = F.embedding(indices, codebook)

                recon_loss = F.mse_loss(quantized, transformed)
                ortho_loss = self.layers[layer_idx].transform.orthogonality_loss()

                loss = recon_loss + self.orthogonality_reg * ortho_loss
                loss.backward()
                self.optimizer.step()

                total_loss += recon_loss.item()
                num_batches += 1
                num_batches_layer += 1

                if num_batches_layer >= num_batches_per_layer:
                    break

        return total_loss / max(num_batches, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            'layers': self.layers.state_dict(),
            'layer_sizes': self.layer_sizes,
            'codebook_dim': self.codebook_dim
        }, path)


class GemmaWeightLoader:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device

        with open(self.model_dir / 'config.json') as f:
            config = json.load(f)

        self.safetensor_path = self.model_dir / 'model.safetensors'
        with open(self.safetensor_path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            self.header = json.loads(f.read(header_size))

        self.weight_keys = []
        for key, info in self.header.items():
            if key == '__metadata__':
                continue
            if 'weight' not in key or len(info['shape']) != 2:
                continue
            if 'lm_head' in key or 'embed_tokens' in key or 'norm' in key:
                continue
            if 'audio_tower' in key or 'vision_tower' in key or 'embed_vision' in key:
                continue
            if 'language_model' not in key:
                continue
            self.weight_keys.append(key)

        print(f"Found {len(self.weight_keys)} weights")

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for key in self.weight_keys:
            tensor = self.load_tensor(key)
            yield key, tensor

    def load_tensor(self, key: str) -> torch.Tensor:
        info = self.header[key]
        begin, end = info['data_offsets']
        numpy_dtype = {
            'F16': np.float16, 'BF16': np.float16, 'F32': np.float32,
        }.get(info['dtype'], np.float32)

        with open(self.safetensor_path, 'rb') as f:
            f.seek(8 + struct.calcsize('Q') + begin)
            data = f.read(end - begin)

        return torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape'])

    def get_layer_info(self) -> List[Tuple[int, int]]:
        layer_sizes = []
        for key in self.weight_keys:
            info = self.header[key]
            layer_sizes.append((info['shape'][0], info['shape'][1]))
        return layer_sizes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train learnable transforms on Gemma weights")
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--output', default='quantized/gemma_transforms.pt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batches-per-layer', type=int, default=100)
    parser.add_argument('--codebook-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--orthogonality-reg', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    print("Initializing weight loader...")
    loader = GemmaWeightLoader(args.model_dir, device)
    layer_sizes = loader.get_layer_info()
    print(f"Layer sizes count: {len(layer_sizes)}")

    print("\nInitializing trainer...")
    trainer = BTCLLMTrainer(
        layer_sizes=layer_sizes,
        codebook_dim=args.codebook_dim,
        lr=args.lr,
        orthogonality_reg=args.orthogonality_reg,
        device=device
    )

    def weight_iter():
        for key, tensor in loader:
            yield tensor.to(device)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(weight_iter(), num_batches_per_layer=args.batches_per_layer)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.6f}")

    print("\nSaving transforms...")
    trainer.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()