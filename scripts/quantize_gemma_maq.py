import torch
import numpy as np
from typing import Dict, Tuple, List
import json
import struct
from pathlib import Path
import os


def magnitude_aware_quantize(w: torch.Tensor, num_bits: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weight magnitudes to num_bits, keep sign as binary"""
    # w shape: (out_features, in_features)
    # Quantize column-wise (per-input-dim) for better quality

    orig_shape = w.shape
    w_flat = w.view(-1)

    # Compute per-element magnitude scale
    abs_w = w_flat.abs()
    max_abs = abs_w.max()

    # Non-uniform quantization for magnitude based on distribution
    # Use log-scale quantization for better handling of outliers
    log_abs = torch.log1p(abs_w)
    log_max = torch.log1p(max_abs)

    # Quantize log-magnitudes
    qmax = 2 ** (num_bits - 1) - 1
    scale = log_max / qmax
    log_q = (log_abs / scale).round().clamp(-qmax, qmax)
    mag_q = log_q * scale

    # Recover quantized magnitude
    quant_mag = torch.expm1(mag_q)

    # Sign is binary (±1)
    sign_q = torch.sign(w_flat)
    sign_q[sign_q == 0] = 1

    # Final: sign * magnitude
    quantized = sign_q * quant_mag

    return quantized.view(orig_shape), scale, max_abs


def per_channel_quantize(w: torch.Tensor, num_bits: int = 4) -> Tuple[torch.Tensor, List]:
    """Per-output-channel quantization (row-wise)"""
    orig_shape = w.shape
    w = w.view(w.shape[0], -1)

    scales = []
    quantized_rows = []

    for i in range(w.shape[0]):
        row = w[i]
        max_val = row.abs().max()
        scale = max_val / (2 ** (num_bits - 1) - 1)
        q_row = (row / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
        scales.append(scale.item())
        quantized_rows.append(q_row)

    return torch.stack(quantized_rows).view(orig_shape), scales


def per_tensor_quantize(w: torch.Tensor, num_bits: int = 2) -> Tuple[torch.Tensor, float]:
    """Simple per-tensor quantization"""
    max_val = w.abs().max()
    scale = max_val / (2 ** (num_bits - 1) - 1) if max_val > 0 else 1.0
    q = (w / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
    return q, scale.item()


def block_quantize(w: torch.Tensor, block_size: int = 128, num_bits: int = 2) -> Tuple[torch.Tensor, List, List]:
    """Block-wise quantization for better quality"""
    orig_shape = w.shape
    h, w_dim = orig_shape

    # Reshape into blocks
    h_blocks = h // block_size
    w_blocks = w_dim // block_size

    # Pad if needed
    pad_h = (h_blocks + 1) * block_size - h if h % block_size != 0 else 0
    pad_w = (w_blocks + 1) * block_size - w_dim if w_dim % block_size != 0 else 0

    if pad_h > 0 or pad_w > 0:
        w = torch.nn.functional.pad(w, (0, pad_w, 0, pad_h))

    new_h, new_w = w.shape
    h_blocks = new_h // block_size
    w_blocks = new_w // block_size

    scales = []
    quantized_blocks = []

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = w[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            max_val = block.abs().max()
            scale = max_val / (2 ** (num_bits - 1) - 1) if max_val > 0 else 1.0
            q_block = (block / scale).round().clamp(-(2**(num_bits-1)), 2**(num_bits-1)-1)
            scales.append(scale.item())
            quantized_blocks.append(q_block)

    # Reshape back
    quantized = torch.stack(quantized_blocks, dim=0).view(h_blocks, w_blocks, block_size, block_size)
    # Permute to (h, w, block, block) then reshape
    quantized = quantized.permute(0, 2, 1, 3).contiguous().view(new_h, new_w)
    quantized = quantized[:orig_shape[0], :orig_shape[1]]

    return quantized, scales, (h_blocks, w_blocks)


class GemmaQuantizer:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.weight_data = {}

    def load_weights(self, max_layers: int = None):
        """Load language model weights only"""
        print("Loading weights...")

        safetensor_path = self.model_dir / 'model.safetensors'
        with open(safetensor_path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_size))

        idx = 0
        for key, info in header.items():
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

            begin, end = info['data_offsets']
            numpy_dtype = {
                'F16': np.float16, 'BF16': np.float16, 'F32': np.float32,
            }.get(info['dtype'], np.float32)

            with open(safetensor_path, 'rb') as f:
                f.seek(8 + header_size + begin)
                data = f.read(end - begin)

            tensor = torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape'])
            self.weight_data[idx] = (key, tensor.float())
            idx += 1

            if max_layers and idx >= max_layers:
                break

        print(f"Loaded {len(self.weight_data)} weight matrices")

    def quantize_with_rank_adaptive_svd(self, energy_threshold: float = 0.80, max_rank: int = 256) -> Dict:
        """Quantize using rank-adaptive SVD with lower threshold for actual compression"""
        print(f"\nRank-adaptive SVD quantization (threshold={energy_threshold}, max_rank={max_rank})")

        from src.lowrank_factorization import compute_optimal_rank, low_rank_factorize
        from src.quantization import quantize_factor, pack_factor

        quantized = {}
        stats = {
            'total_original': 0,
            'total_stored': 0,
            'layer_stats': []
        }

        for idx in sorted(self.weight_data.keys()):
            key, W = self.weight_data[idx]
            W_float = W.float()

            orig_params = W_float.shape[0] * W_float.shape[1]
            stats['total_original'] += orig_params

            # Compute optimal rank
            U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)
            energy = (S ** 2).cumsum(dim=0) / (S ** 2).sum()
            rank = (energy < energy_threshold).sum().item() + 1
            rank = min(rank, max_rank)

            # If rank is too high, use max_rank with truncated SVD
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vt_r = Vt[:rank, :]

            # Quantize
            q_data = quantize_factor(U_r, S_r, Vt_r, sigma_bits=2)
            p_data = pack_factor(q_data)
            p_data['rank'] = rank
            p_data['original_shape'] = list(W_float.shape)
            p_data['key'] = key
            p_data['energy_captured'] = (S_r ** 2).sum().item() / (S ** 2).sum().item()

            # Calculate stored params
            stored_params = U_r.numel() + S_r.numel() + Vt_r.numel()
            stats['total_stored'] += stored_params

            # Calculate actual bits/weight
            u_bits = U_r.numel() * 0.625  # ternary
            s_bits = S_r.numel() * 2      # 2-bit sigma
            vt_bits = Vt_r.numel() * 0.625
            total_bits = u_bits + s_bits + vt_bits
            bpw = total_bits / orig_params

            stats['layer_stats'].append({
                'idx': idx,
                'key': key[:50],
                'shape': list(W_float.shape),
                'rank': rank,
                'bpw': bpw,
                'energy': p_data['energy_captured']
            })

            quantized[idx] = p_data

            if idx % 50 == 0 or idx < 5:
                print(f"  Layer {idx}: rank={rank}, bpw={bpw:.4f}, shape={list(W_float.shape)}")

        # Calculate overall stats
        total_bits = sum(
            ls['rank'] * (ls['shape'][0] + ls['shape'][1] + 1) * 0.625 + ls['rank'] * 2
            for ls in stats['layer_stats']
        )
        stats['avg_bpw'] = total_bits / stats['total_original']
        stats['compression'] = stats['total_original'] * 16 / total_bits

        return quantized, stats

    def quantize_binary_magnitude(self, num_mag_bits: int = 2) -> Dict:
        """Binary sign + magnitude quantization for maximum compression"""
        print(f"\nBinary Magnitude Quantization ({num_mag_bits}-bit magnitudes)")

        quantized = {}
        stats = {'total_original': 0, 'total_stored': 0}

        for idx in sorted(self.weight_data.keys()):
            key, W = self.weight_data[idx]
            W_float = W.float()

            orig_params = W_float.shape[0] * W_float.shape[1]
            stats['total_original'] += orig_params

            # Sign as binary (±1 encoded as 0/1)
            sign = torch.sign(W_float)
            sign[sign == 0] = 1
            sign_binary = (sign + 1) // 2  # Convert -1→0, 1→1

            # Magnitude quantized per-tensor
            mag = W_float.abs()
            max_mag = mag.max()
            if max_mag > 0:
                scale = max_mag / (2 ** (num_mag_bits - 1) - 1)
                mag_q = (mag / scale).round().clamp(0, 2 ** (num_mag_bits - 1) - 1)
            else:
                scale = 1.0
                mag_q = torch.zeros_like(mag)

            # Reconstruct
            sign_recon = sign_binary * 2 - 1  # Convert back to ±1
            W_recon = sign_recon * mag_q * scale

            # Calculate compression
            sign_bits = sign_binary.numel()  # 1 bit
            mag_bits = mag_q.numel() * num_mag_bits
            scale_bits = 16  # scale as float16
            total_bits = sign_bits + mag_bits + scale_bits
            bpw = total_bits / orig_params

            stored_params = total_bits / 16  # Convert to 16-bit equivalent

            stats['total_stored'] += stored_params

            quantized[idx] = {
                'sign_binary': sign_binary.to(torch.uint8),
                'mag_q': mag_q.to(torch.uint8),
                'scale': scale.item(),
                'shape': list(W_float.shape),
                'key': key,
                'bpw': bpw,
                'num_mag_bits': num_mag_bits
            }

            if idx % 50 == 0 or idx < 5:
                print(f"  Layer {idx}: bpw={bpw:.4f}, shape={list(W_float.shape)}")

        stats['avg_bpw'] = sum(q['bpw'] for q in quantized.values()) / len(quantized)
        stats['compression'] = stats['total_original'] * 16 / (stats['total_stored'] * 16)

        return quantized, stats

    def quantize_hybrid(self, importance_scores: Dict[int, float] = None, default_bits: int = 2) -> Dict:
        """Hybrid quantization: important layers get more bits"""
        print(f"\nHybrid Quantization (important={default_bits+2}bit, normal={default_bits}bit)")

        quantized = {}
        stats = {'total_original': 0, 'total_stored': 0, 'bits分配': {}}

        if importance_scores is None:
            # Default: early layers (0-10) and full-attention layers are more important
            importance_scores = {}
            for idx in sorted(self.weight_data.keys()):
                key, _ = self.weight_data[idx]
                # Layer index from key
                layer_num = int(key.split('.layers.')[1].split('.')[0]) if 'layers.' in key else idx
                is_full_attn = 'full_attention' in key.lower() if key else False

                # Early layers and full attention get more precision
                if layer_num < 5:
                    importance_scores[idx] = 4  # Important
                elif is_full_attn:
                    importance_scores[idx] = 4
                else:
                    importance_scores[idx] = default_bits

        for idx in sorted(self.weight_data.keys()):
            key, W = self.weight_data[idx]
            W_float = W.float()

            orig_params = W_float.shape[0] * W_float.shape[1]
            stats['total_original'] += orig_params

            num_bits = importance_scores.get(idx, default_bits)

            # Per-tensor quantization
            max_val = W_float.abs().max()
            scale = max_val / (2 ** (num_bits - 1) - 1) if max_val > 0 else 1.0
            q = (W_float / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)

            # Calculate bits
            total_bits = q.numel() * num_bits + 16  # +16 for scale
            bpw = total_bits / orig_params

            stats['total_stored'] += total_bits / 16
            if num_bits not in stats['bits分配']:
                stats['bits分配'][num_bits] = 0
            stats['bits分配'][num_bits] += orig_params

            quantized[idx] = {
                'q': q.to(torch.int8),
                'scale': scale.item(),
                'num_bits': num_bits,
                'shape': list(W_float.shape),
                'key': key,
                'bpw': bpw
            }

            if idx % 50 == 0 or idx < 5:
                print(f"  Layer {idx}: {num_bits}-bit, bpw={bpw:.4f}, shape={list(W_float.shape)}")

        stats['avg_bpw'] = sum(q['bpw'] for q in quantized.values()) / len(quantized)
        stats['compression'] = stats['total_original'] * 16 / stats['total_stored']

        return quantized, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quantize Gemma with improved methods")
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--output', default='quantized/gemma_maq.pt')
    parser.add_argument('--method', choices=['svd', 'binary', 'hybrid'], default='hybrid')
    parser.add_argument('--energy-threshold', type=float, default=0.80)
    parser.add_argument('--max-rank', type=int, default=256)
    parser.add_argument('--mag-bits', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    quantizer = GemmaQuantizer(args.model_dir, args.device)
    quantizer.load_weights()

    if args.method == 'svd':
        quantized, stats = quantizer.quantize_with_rank_adaptive_svd(
            energy_threshold=args.energy_threshold,
            max_rank=args.max_rank
        )
    elif args.method == 'binary':
        quantized, stats = quantizer.quantize_binary_magnitude(num_mag_bits=args.mag_bits)
    else:  # hybrid
        quantized, stats = quantizer.quantize_hybrid()

    print(f"\n=== Results ===")
    print(f"Average bits/weight: {stats['avg_bpw']:.4f}")
    print(f"Compression vs FP16: {stats['compression']:.1f}x")

    if 'bits分配' in stats:
        print(f"Bits allocation: {stats['bits分配']}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        'quantized': quantized,
        'stats': stats,
        'method': args.method,
        'config': {
            'energy_threshold': args.energy_threshold,
            'max_rank': args.max_rank,
            'mag_bits': args.mag_bits
        }
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()