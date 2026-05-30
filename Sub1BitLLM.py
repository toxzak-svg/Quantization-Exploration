import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, List, Iterator, Tuple
from pathlib import Path
import os
import numpy as np

from src.gguf_writer import GGUFWriter, GGML_TYPES


@dataclass
class Sub1BitConfig:
    codebook_dim: int = 128
    energy_threshold: float = 0.95
    rank: int = 16
    U_bits: float = 0.5
    S_bits: float = 2.0
    Vt_bits: float = 0.5
    model_name: str = "llama-2-7b-sub1bit"
    architecture: str = "llama"


class LowRankFactor(torch.nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor):
        super().__init__()
        self.register_buffer('U', U.half())
        self.register_buffer('S', S.half())
        self.register_buffer('Vt', Vt.half())
        self.rank = U.shape[1]

    def forward(self) -> torch.Tensor:
        return torch.matmul(self.U * self.S.unsqueeze(0), self.Vt)

    def forward_lowrank(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(x, self.Vt.T) * self.S.unsqueeze(0), self.U.T)


class TernaryQuantizedFactor(torch.nn.Module):
    def __init__(self, data: torch.Tensor, scale: torch.Tensor, rank: int):
        super().__init__()
        self.register_buffer('data', data)
        self.register_buffer('scale', scale)
        self.rank = rank

    def forward(self) -> torch.Tensor:
        return self.data.float() * self.scale


class Sub1BitLLM(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        config: Optional[Sub1BitConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.model_path = model_path
        self.config = config or Sub1BitConfig()
        self.device = device
        self.layers: Dict[int, LowRankFactor] = {}
        self.metadata: Dict = {}

    @classmethod
    def from_fp16(
        cls,
        model_path: str,
        config: Optional[Sub1BitConfig] = None,
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> "Sub1BitLLM":
        instance = cls(model_path, config, device)
        if checkpoint_dir is None:
            checkpoint_dir = Path(model_path).parent / "checkpoints"
        else:
            checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        for ckpt_file in sorted(checkpoint_dir.glob("layer_*.pt")):
            layer_idx = int(ckpt_file.stem.split("_")[1])
            factor = torch.load(ckpt_file, weights_only=False, map_location=device)
            U = torch.from_numpy(factor['U']).to(device)
            S = torch.from_numpy(factor['S']).to(device)
            Vt = torch.from_numpy(factor['Vt']).to(device)
            instance.layers[layer_idx] = LowRankFactor(U, S, Vt)
        instance.metadata = {
            'num_layers': len(instance.layers),
            'rank': instance.config.rank,
            'energy_threshold': instance.config.energy_threshold
        }
        return instance

    def load_checkpoint(self, checkpoint_path: str) -> "Sub1BitLLM":
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'layers' in checkpoint:
            for layer_idx, factor_data in checkpoint['layers'].items():
                self.layers[int(layer_idx)] = LowRankFactor(
                    torch.from_numpy(factor_data['U']).to(self.device),
                    torch.from_numpy(factor_data['S']).to(self.device),
                    torch.from_numpy(factor_data['Vt']).to(self.device),
                )
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for layer_idx, layer in self.layers.items():
            state[f'layers.{layer_idx}.U'] = layer.U
            state[f'layers.{layer_idx}.S'] = layer.S
            state[f'layers.{layer_idx}.Vt'] = layer.Vt
        return state

    def forward(self, x: torch.Tensor, layer_indices: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        outputs = {}
        indices = layer_indices if layer_indices is not None else list(self.layers.keys())
        for idx in indices:
            if idx in self.layers:
                outputs[idx] = self.layers[idx].forward_lowrank(x)
        return outputs

    def get_weight(self, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self.layers:
            raise KeyError(f"Layer {layer_idx} not found")
        return self.layers[layer_idx]()

    def iter_layers(self) -> Iterator[Tuple[int, LowRankFactor]]:
        for idx in sorted(self.layers.keys()):
            yield idx, self.layers[idx]

    def compression_stats(self) -> Dict[str, float]:
        total_original = 0
        total_factor = 0
        for _, layer in self.iter_layers():
            orig_size = layer.U.shape[0] * layer.Vt.shape[1]
            factor_size = layer.U.numel() + layer.S.numel() + layer.Vt.numel()
            total_original += orig_size
            total_factor += factor_size
        return {
            'compression_ratio': total_original / total_factor if total_factor > 0 else 0,
            'avg_rank': sum(l.rank for _, l in self.iter_layers()) / max(len(self.layers), 1)
        }

    def to_gguf(self, output_path: str, metadata: Optional[Dict] = None):
        writer = GGUFWriter(output_path)
        writer.add_key_value("general.architecture", self.config.architecture)
        writer.add_key_value("general.name", self.config.model_name)
        writer.add_key_value("quantization.type", "sub1bit_lowrank")
        writer.add_key_value("quantization.U_bits", self.config.U_bits)
        writer.add_key_value("quantization.S_bits", self.config.S_bits)
        writer.add_key_value("quantization.Vt_bits", self.config.Vt_bits)
        if metadata:
            for key, value in metadata.items():
                writer.add_key_value(key, value)
        for layer_idx, layer in self.iter_layers():
            writer.add_tensor(
                f"model.layers.{layer_idx}.U",
                layer.U.cpu().numpy().astype(np.float16),
                GGML_TYPES['float16']
            )
            writer.add_tensor(
                f"model.layers.{layer_idx}.S",
                layer.S.cpu().numpy().astype(np.float16),
                GGML_TYPES['float16']
            )
            writer.add_tensor(
                f"model.layers.{layer_idx}.Vt",
                layer.Vt.cpu().numpy().astype(np.float16),
                GGML_TYPES['float16']
            )
            writer.add_tensor(
                f"model.layers.{layer_idx}.rank",
                np.array([layer.rank], dtype=np.int32),
                GGML_TYPES['int32']
            )
        writer.write()
        return os.path.getsize(output_path)


def from_fp16(
    model_path: str,
    config: Optional[Sub1BitConfig] = None,
    checkpoint_dir: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Sub1BitLLM:
    return Sub1BitLLM.from_fp16(model_path, config, checkpoint_dir, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sub1BitLLM API Demo")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    config = Sub1BitConfig(
        codebook_dim=128,
        energy_threshold=0.95,
        model_name="llama-2-7b-sub1bit"
    )
    model = from_fp16(args.model, config=config, checkpoint_dir=args.checkpoint_dir, device=args.device)
    print(f"Loaded Sub1BitLLM with {len(model.layers)} layers")
    print(f"Compression stats: {model.compression_stats()}")
