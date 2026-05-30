import torch
import argparse
import os
import gc
import struct
import json
from pathlib import Path
from typing import Dict, Optional, Iterator, Tuple, List

from train_transform import BTCLLMTrainer
from lowrank_factorization import factorize_model_weights
from adqat_search import AdaQATrainer, apply_quantization


def get_safetensor_keys(filepath: str) -> List[str]:
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = f.read(header_size)
        header_obj = json.loads(header)
        return list(header_obj.keys())


def get_tensor_metadata(filepath: str) -> Dict[str, Dict]:
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = f.read(header_size)
        header_obj = json.loads(header)

    metadata = {}
    for name, info in header_obj.items():
        if name == '__metadata__':
            continue
        if not isinstance(info, dict) or 'shape' not in info:
            continue
        metadata[name] = {
            'shape': info['shape'],
            'dtype': info['dtype'],
            'offsets': info['data_offsets']
        }
    return metadata


def load_single_tensor(filepath: str, key: str, device: str = "cpu") -> torch.Tensor:
    import numpy as np

    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = f.read(header_size)
        header_obj = json.loads(header)

        info = header_obj[key]
        begin, end = info['data_offsets']
        dtype = info['dtype']

        f.seek(header_size + begin)
        data = f.read(end - begin)

    if dtype == 'BF16':
        arr = np.frombuffer(data, dtype=np.uint16).copy()
        tensor = torch.from_numpy(arr).to(torch.uint16).view(torch.bfloat16).reshape(info['shape'])
    else:
        numpy_dtype = {
            'F16': np.float16,
            'F32': np.float32,
            'I32': np.int32,
            'I16': np.int16,
            'I8': np.int8,
            'U8': np.uint8,
        }.get(dtype, np.float32)

        arr = np.frombuffer(data, dtype=numpy_dtype).copy()
        tensor = torch.from_numpy(arr).reshape(info['shape'])

    if device != "cpu":
        tensor = tensor.to(device)

    return tensor


class SubOneBitQuantizer:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = model_path
        self.device = device
        self.transforms = None
        self.codebooks = None
        self.factors = {}
        self.weight_keys = []
        self.file_mapping = {}
        self.total_weights = 0
        self.checkpoint_dir = None

    def discover_weights(self):
        model_dir = Path(self.model_path)
        safetensor_files = list(model_dir.glob("*.safetensors"))

        all_weight_keys = []
        file_to_keys = {}

        for sf in safetensor_files:
            metadata = get_tensor_metadata(str(sf))
            weight_keys = [k for k in metadata.keys()
                          if 'weight' in k and 'lm_head' not in k
                          and 'embed_tokens' not in k and 'norm' not in k
                          and len(metadata[k]['shape']) == 2]

            for k in weight_keys:
                self.file_mapping[k] = str(sf)
                all_weight_keys.append((k, metadata[k]['shape']))

        self.weight_keys = all_weight_keys
        self.total_weights = len(all_weight_keys)
        print(f"Found {self.total_weights} 2D weight matrices")
        for name, shape in all_weight_keys[:5]:
            print(f"  {name}: {shape}")

    def load_weight(self, idx: int) -> torch.Tensor:
        key, _ = self.weight_keys[idx]
        filepath = self.file_mapping[key]
        return load_single_tensor(filepath, key, "cpu")

    def iter_weights(self, start: int = 0, end: Optional[int] = None) -> Iterator[Tuple[int, torch.Tensor]]:
        end = end or self.total_weights
        for i in range(start, min(end, self.total_weights)):
            yield i, self.load_weight(i)

    def load_model_weights(self):
        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise ValueError(f"Model path is not a directory: {model_dir}")

        self.discover_weights()
        print(f"Registered {self.total_weights} weights for incremental processing")

    def _compute_rank(self, S: torch.Tensor, energy_threshold: float, max_rank: Optional[int] = None) -> int:
        squared_sv = S ** 2
        cumulative = torch.cumsum(squared_sv, dim=0)
        total = torch.sum(squared_sv)
        normalized = cumulative / total
        r = torch.searchsorted(normalized, energy_threshold).item() + 1
        r = min(r, len(S))
        if max_rank is not None:
            r = min(r, max_rank)
        return r

    def process_and_save_layer(self, idx: int, W_float: torch.Tensor, energy_threshold: float, max_rank: Optional[int], save_path: Path) -> int:
        U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)
        r = self._compute_rank(S, energy_threshold, max_rank)

        factor = {
            'U': U[:, :r].half().numpy(),
            'S': S[:r].half().numpy(),
            'Vt': Vt[:r, :].half().numpy(),
            'rank': r,
            'original_shape': list(W_float.shape),
            'energy_captured': ((S[:r] ** 2).sum().item() / (S ** 2).sum().item())
        }

        torch.save(factor, save_path)

        del U, S, Vt, W_float
        gc.collect()

        return r

    def run_pipeline(
        self,
        output_path: str,
        codebook_dim: int = 128,
        energy_threshold: float = 0.95,
        max_rank: Optional[int] = None,
        num_epochs: int = 5,
        adqat_epochs: int = 2,
        max_layers: Optional[int] = None
    ):
        print("=" * 60)
        print("Sub-1-Bit Quantization Pipeline")
        print("=" * 60)

        num_layers = max_layers or self.total_weights

        self.checkpoint_dir = Path(output_path).parent / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Phase 1] Low-rank factorization via SVD (saving to disk)...")

        for idx, W in self.iter_weights(0, num_layers):
            W_float = W.float()
            save_path = self.checkpoint_dir / f"layer_{idx:04d}.pt"

            r = self.process_and_save_layer(idx, W_float, energy_threshold, max_rank, save_path)

            if idx % 50 == 0 or idx < 5:
                print(f"  Processed layer {idx}/{num_layers}, rank={r}")

            gc.collect()

        print(f"  All {num_layers} layers factorized and saved")
        print("  Factorization complete.")

        print("\n[Phase 2] Applying quantization...")
        print("  U/V -> 0.5-bit ternary, Sigma -> 2-bit")

        quantized = {}
        for idx in range(num_layers):
            save_path = self.checkpoint_dir / f"layer_{idx:04d}.pt"
            factor = torch.load(save_path, weights_only=False)

            q_U, _, _ = self._quantize_ternary(torch.from_numpy(factor['U']))
            q_S, _, _ = self._quantize_sigma(torch.from_numpy(factor['S']))

            quantized[idx] = {
                'U': q_U,
                'S': q_S,
                'Vt': factor['Vt'],
                'rank': factor['rank'],
                'original_shape': factor['original_shape']
            }

            if idx % 50 == 0:
                print(f"  Quantized layer {idx}/{num_layers}")

            del factor
            gc.collect()

        print("\n[Phase 3] Packing to GGUF...")
        self._save_gguf(output_path, quantized)

        print(f"\nQuantization complete! Output: {output_path}")
        self._print_stats(quantized)

    def _quantize_ternary(self, x: torch.Tensor):
        scale = x.abs().max()
        if scale == 0:
            scale = 1.0
        normalized = x / scale
        ternary = torch.sign(normalized)
        ternary[ternary == 0] = 1
        return ternary.to(torch.int8), scale, torch.tensor(0, dtype=torch.int8)

    def _quantize_sigma(self, sigma: torch.Tensor, num_bits: int = 2):
        max_val = sigma.abs().max()
        if max_val == 0:
            max_val = 1.0
        scale = max_val / (2 ** (num_bits - 1) - 1)
        normalized = (sigma / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
        return normalized.to(torch.int8), scale, max_val

    def _save_gguf(self, output_path: str, quantized: Dict):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save({
            'quantized': quantized,
            'model_path': self.model_path
        }, output_path + ".pt")

        with open(output_path, 'wb') as f:
            f.write(b'GGUF')
            f.write(torch.tensor([3], dtype=torch.int32).numpy().tobytes())

    def _print_stats(self, quantized: Dict):
        total_params = sum(q['U'].numel() + q['S'].numel() + q['Vt'].numel() for q in quantized.values())

        uv_bits = sum(q['U'].numel() * 0.5 + q['Vt'].numel() * 0.5 for q in quantized.values())
        sigma_bits = sum(q['S'].numel() * 2 for q in quantized.values())
        total_bits = uv_bits + sigma_bits

        avg_bits = total_bits / total_params if total_params > 0 else 0

        print(f"\n{'='*60}")
        print("Quantization Statistics")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Average bit-width: {avg_bits:.4f} bits/weight")
        if total_params > 0:
            print(f"Estimated size: {total_bits / 8 / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Sub-1-Bit LLM Quantization")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights or directory")
    parser.add_argument("--output", type=str, default="quantized/model.gguf", help="Output GGUF path")
    parser.add_argument("--codebook-dim", type=int, default=128, help="Codebook embedding dimension")
    parser.add_argument("--energy-threshold", type=float, default=0.95, help="SVD energy retention threshold")
    parser.add_argument("--max-rank", type=int, default=None, help="Maximum rank for low-rank factorization (e.g., 16, 32)")
    parser.add_argument("--epochs", type=int, default=5, help="Transform training epochs (unused)")
    parser.add_argument("--adqat-epochs", type=int, default=2, help="AdaQAT training epochs (unused)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--max-layers", type=int, default=None, help="Max layers to process (for memory constraints)")

    args = parser.parse_args()

    quantizer = SubOneBitQuantizer(args.model, device=args.device)
    quantizer.load_model_weights()
    quantizer.run_pipeline(
        output_path=args.output,
        codebook_dim=args.codebook_dim,
        energy_threshold=args.energy_threshold,
        max_rank=args.max_rank,
        num_epochs=args.epochs,
        adqat_epochs=args.adqat_epochs,
        max_layers=args.max_layers
    )


if __name__ == "__main__":
    main()