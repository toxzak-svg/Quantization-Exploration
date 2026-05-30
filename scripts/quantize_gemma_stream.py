import torch
import numpy as np
from typing import Dict, Tuple, List, Iterator
import json
import struct
from pathlib import Path
import os
import gc


class GemmaStreamingQuantizer:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.weight_info = []

        with open(self.model_dir / 'config.json') as f:
            config = json.load(f)

        self.safetensor_path = self.model_dir / 'model.safetensors'
        with open(self.safetensor_path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            self.header = json.loads(f.read(header_size))

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
            self.weight_info.append((key, info))

        print(f"Found {len(self.weight_info)} weights to quantize")

    def load_tensor_raw(self, key: str, info: dict) -> torch.Tensor:
        begin, end = info['data_offsets']
        numpy_dtype = {
            'F16': np.float16, 'BF16': np.float16, 'F32': np.float32,
        }.get(info['dtype'], np.float32)

        with open(self.safetensor_path, 'rb') as f:
            f.seek(8 + struct.calcsize('Q') + begin)
            data = f.read(end - begin)

        return torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape'])

    def stream_weights(self) -> Iterator[Tuple[int, str, torch.Tensor]]:
        for idx, (key, info) in enumerate(self.weight_info):
            W = self.load_tensor_raw(key, info).float()
            yield idx, key, W
            del W
            gc.collect()

    def quantize_hybrid_streaming(self, importance_fn=None) -> Dict:
        """Streaming hybrid quantization - doesn't store all weights in memory"""
        if importance_fn is None:
            # Default: early layers and full attention layers get more bits
            def importance_fn(idx, key):
                layer_num = int(key.split('.layers.')[1].split('.')[0]) if 'layers.' in key else idx
                is_full_attn = 'full_attention' in key.lower() if key else False
                if layer_num < 5:
                    return 4
                elif is_full_attn:
                    return 4
                else:
                    return 2

        print("\nStreaming hybrid quantization...")
        quantized = {}
        stats = {'total_original': 0, 'total_bits': 0, 'layer_count': 0}
        bits_distribution = {2: 0, 3: 0, 4: 0, 6: 0}

        for idx, key, W in self.stream_weights():
            W_float = W.float()
            orig_params = W_float.shape[0] * W_float.shape[1]
            stats['total_original'] += orig_params

            num_bits = importance_fn(idx, key)
            bits_distribution[num_bits] = bits_distribution.get(num_bits, 0) + orig_params

            max_val = W_float.abs().max()
            scale = max_val / (2 ** (num_bits - 1) - 1) if max_val > 0 else 1.0
            q = (W_float / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)

            total_bits = q.numel() * num_bits + 16  # +16 for scale
            stats['total_bits'] += total_bits

            quantized[idx] = {
                'q': q.to(torch.int8),
                'scale': scale.item(),
                'num_bits': num_bits,
                'shape': list(W_float.shape),
                'key': key
            }

            stats['layer_count'] += 1
            if idx % 50 == 0 or idx < 5:
                bpw = total_bits / orig_params
                print(f"  Layer {idx}: {num_bits}-bit, bpw={bpw:.4f}")

            del W, W_float
            gc.collect()

        stats['avg_bpw'] = stats['total_bits'] / stats['total_original']
        stats['compression'] = stats['total_original'] * 16 / stats['total_bits']
        stats['bits_distribution'] = bits_distribution

        return quantized, stats

    def quantize_binary_magnitude_streaming(self, num_mag_bits: int = 2) -> Dict:
        """Binary sign + magnitude quantization"""
        print(f"\nStreaming Binary Magnitude Quantization ({num_mag_bits}-bit magnitudes)...")

        quantized = {}
        stats = {'total_original': 0, 'total_bits': 0}

        for idx, key, W in self.stream_weights():
            W_float = W.float()
            orig_params = W_float.shape[0] * W_float.shape[1]
            stats['total_original'] += orig_params

            sign = torch.sign(W_float)
            sign[sign == 0] = 1
            sign_binary = (sign + 1) // 2

            mag = W_float.abs()
            max_mag = mag.max()
            if max_mag > 0:
                scale = max_mag / (2 ** (num_mag_bits - 1) - 1)
                mag_q = (mag / scale).round().clamp(0, 2 ** (num_mag_bits - 1) - 1)
            else:
                scale = 1.0
                mag_q = torch.zeros_like(mag)

            total_bits = sign_binary.numel() + mag_q.numel() * num_mag_bits + 16
            stats['total_bits'] += total_bits

            quantized[idx] = {
                'sign': sign_binary.to(torch.uint8),
                'mag': mag_q.to(torch.uint8),
                'scale': scale.item(),
                'num_mag_bits': num_mag_bits,
                'shape': list(W_float.shape),
                'key': key
            }

            if idx % 50 == 0 or idx < 5:
                bpw = total_bits / orig_params
                print(f"  Layer {idx}: bpw={bpw:.4f}")

            del W, W_float
            gc.collect()

        stats['avg_bpw'] = stats['total_bits'] / stats['total_original']
        stats['compression'] = stats['total_original'] * 16 / stats['total_bits']

        return quantized, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stream quantize Gemma")
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--output', default='quantized/gemma_hybrid.pt')
    parser.add_argument('--method', choices=['hybrid', 'binary'], default='hybrid')
    parser.add_argument('--mag-bits', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    quantizer = GemmaStreamingQuantizer(args.model_dir, args.device)

    if args.method == 'hybrid':
        quantized, stats = quantizer.quantize_hybrid_streaming()
    else:
        quantized, stats = quantizer.quantize_binary_magnitude_streaming(num_mag_bits=args.mag_bits)

    print(f"\n=== Results ===")
    print(f"Layers processed: {stats['layer_count']}")
    print(f"Average bits/weight: {stats['avg_bpw']:.4f}")
    print(f"Compression vs FP16: {stats['compression']:.1f}x")

    if 'bits_distribution' in stats:
        print(f"Bits distribution: {stats['bits_distribution']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        'quantized': quantized,
        'stats': stats,
        'method': args.method
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()