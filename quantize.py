import torch
import gc
import json
import struct
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Iterator, Tuple, List

from src.lowrank_factorization import compute_optimal_rank, low_rank_factorize
from src.quantization import quantize_factor, pack_factor


def get_safetensor_keys(filepath: str) -> List[str]:
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = f.read(header_size)
        return list(json.loads(header).keys())


def get_tensor_metadata(filepath: str) -> Dict[str, Dict]:
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = f.read(header_size)
        header_obj = json.loads(header)
    metadata = {}
    for name, info in header_obj.items():
        if name == '__metadata__' or not isinstance(info, dict) or 'shape' not in info:
            continue
        metadata[name] = {
            'shape': info['shape'],
            'dtype': info['dtype'],
            'offsets': info['data_offsets']
        }
    return metadata


def load_single_tensor(filepath: str, key: str, device: str = "cpu") -> torch.Tensor:
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = f.read(header_size)
        info = json.loads(header)[key]
        begin, end = info['data_offsets']
        f.seek(8 + header_size + begin)
        data = f.read(end - begin)
    numpy_dtype = {
        'F16': np.float16, 'BF16': np.float16, 'F32': np.float32,
        'I32': np.int32, 'I16': np.int16,
        'I8': np.int8, 'U8': np.uint8,
    }.get(info['dtype'], np.float32)
    tensor = torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape'])
    return tensor.to(device) if device != "cpu" else tensor


class SubOneBitQuantizer:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_path = model_path
        self.weight_keys: List[Tuple[str, List[int]]] = []
        self.file_mapping: Dict[str, str] = {}

    def discover_weights(self):
        model_dir = Path(self.model_path)
        safetensor_files = list(model_dir.glob("*.safetensors"))
        for sf in safetensor_files:
            metadata = get_tensor_metadata(str(sf))
            for k, v in metadata.items():
                if 'weight' not in k or len(v['shape']) != 2:
                    continue
                if 'lm_head' in k or 'embed_tokens' in k or 'norm' in k:
                    continue
                if 'audio_tower' in k or 'vision_tower' in k or 'embed_vision' in k:
                    continue
                if 'language_model' not in k:
                    continue
                self.file_mapping[k] = str(sf)
                self.weight_keys.append((k, v['shape']))
        print(f"Found {len(self.weight_keys)} 2D weight matrices")
        for name, shape in self.weight_keys[:5]:
            print(f"  {name}: {shape}")

    def load_weight(self, idx: int) -> torch.Tensor:
        key, _ = self.weight_keys[idx]
        return load_single_tensor(self.file_mapping[key], key, self.device)

    def iter_weights(self, start: int = 0, end: Optional[int] = None) -> Iterator[Tuple[int, torch.Tensor]]:
        end = end or len(self.weight_keys)
        for i in range(start, min(end, len(self.weight_keys))):
            yield i, self.load_weight(i)

    def run_pipeline(
        self,
        output_path: str,
        energy_threshold: float = 0.95,
        max_layers: Optional[int] = None,
        sigma_bits: int = 2,
    ):
        print("=" * 60)
        print("Sub-1-Bit Quantization Pipeline (single-pass GPU)")
        print("=" * 60)

        num_layers = max_layers or len(self.weight_keys)
        packed_factors: Dict[int, Dict] = {}
        total_original_params = 0
        total_stored_params = 0

        for idx, W in self.iter_weights(0, num_layers):
            W_float = W.float()
            U, S, Vt, rank = low_rank_factorize(W_float, energy_threshold)
            q_data = quantize_factor(U, S, Vt, sigma_bits)
            p_data = pack_factor(q_data)
            p_data['rank'] = rank
            p_data['original_shape'] = list(W_float.shape)

            orig_params = W_float.shape[0] * W_float.shape[1]
            stored_params = U.numel() + S.numel() + Vt.numel()
            total_original_params += orig_params
            total_stored_params += stored_params

            packed_factors[idx] = p_data

            if idx % 50 == 0 or idx < 5:
                energy = (S ** 2).sum().item() / (W_float ** 2).sum().item()
                print(f"  Layer {idx}/{num_layers}: rank={rank}, energy={energy:.4f}, shape={list(W_float.shape)}")

            del W, W_float, U, S, Vt
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save({
            'quantized': packed_factors,
            'config': {
                'energy_threshold': energy_threshold,
                'sigma_bits': sigma_bits,
                'model_path': self.model_path,
            }
        }, output_path + ".pt")

        self._print_stats(packed_factors, total_original_params, total_stored_params)
        print(f"\nQuantized model saved to: {output_path}.pt")
        return packed_factors

    def _print_stats(self, packed_factors: Dict, total_original: int, total_stored: int):
        ternary_per_value = 5 / 8
        S_per_value = 2
        total_ternary_bits = sum(
            p['U_packed'].numel() * 8 / 5 * ternary_per_value + p['Vt_packed'].numel() * 8 / 5 * ternary_per_value
            for p in packed_factors.values()
        )
        total_sigma_bits = sum(p['S'].numel() * S_per_value for p in packed_factors.values())
        total_bits = total_ternary_bits + total_sigma_bits
        avg_bit_width = total_bits / total_original if total_original > 0 else 0
        stored_params = sum(
            p['U_packed'].numel() * 8 / 5 + p['Vt_packed'].numel() * 8 / 5 + p['S'].numel()
            for p in packed_factors.values()
        )
        print(f"\n{'='*60}")
        print("Quantization Statistics")
        print(f"{'='*60}")
        print(f"Original parameters: {total_original:,}")
        print(f"Stored parameters (equiv.): {stored_params:,.0f}")
        print(f"Compression ratio (vs FP16): {total_original * 2 / stored_params:.1f}x")
        print(f"Average bit-width: {avg_bit_width:.4f} bits/weight")
        print(f"Estimated size: {total_bits / 8 / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Sub-1-Bit LLM Quantization (single-pass GPU)")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory with safetensors")
    parser.add_argument("--output", type=str, default="quantized/model.gguf", help="Output .gguf.pt path")
    parser.add_argument("--energy-threshold", type=float, default=0.95, help="SVD energy retention (0-1)")
    parser.add_argument("--sigma-bits", type=int, default=2, help="Bits for singular values (1-4)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (auto if omitted)")
    parser.add_argument("--max-layers", type=int, default=None, help="Max layers to process")
    args = parser.parse_args()

    quantizer = SubOneBitQuantizer(args.model, device=args.device)
    quantizer.discover_weights()
    quantizer.run_pipeline(
        output_path=args.output,
        energy_threshold=args.energy_threshold,
        max_layers=args.max_layers,
        sigma_bits=args.sigma_bits,
    )


if __name__ == "__main__":
    main()
