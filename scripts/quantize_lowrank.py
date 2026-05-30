import torch
import numpy as np
from typing import Dict, Tuple, List
import json
import struct
from pathlib import Path
import os
import gc


def lowrank_quantize(W, rank, sigma_bits=2):
    """Lowrank SVD quantization with fixed rank"""
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    # Quantize U and Vt to ternary (0, ±1)
    U_scale = U_r.abs().max()
    if U_scale > 0:
        U_q = (U_r / U_scale).round().clamp(-1, 1).to(torch.int8)
    else:
        U_q = torch.zeros_like(U_r).to(torch.int8)

    Vt_scale = Vt_r.abs().max()
    if Vt_scale > 0:
        Vt_q = (Vt_r / Vt_scale).round().clamp(-1, 1).to(torch.int8)
    else:
        Vt_q = torch.zeros_like(Vt_r).to(torch.int8)

    # Quantize S to 2-bit
    S_scale = S_r.abs().max()
    qmax = 2 ** (sigma_bits - 1) - 1
    if S_scale > 0:
        scale = S_scale / qmax
        S_q = (S_r / scale).round().clamp(-qmax, qmax).to(torch.int8)
    else:
        S_q = torch.zeros_like(S_r).to(torch.int8)
        scale = 1.0

    return {
        'U_q': U_q,
        'U_scale': U_scale.item(),
        'Vt_q': Vt_q,
        'Vt_scale': Vt_scale.item(),
        'S_q': S_q,
        'S_scale': scale.item(),
        'rank': rank
    }


def compute_bits(q_entry, orig_params):
    """Compute bits per weight for a quantized entry"""
    u_bits = q_entry['U_q'].numel() * 1  # ternary = 1 bit
    vt_bits = q_entry['Vt_q'].numel() * 1
    s_bits = q_entry['S_q'].numel() * 2  # 2-bit sigma
    total_bits = u_bits + vt_bits + s_bits + 16 + 16  # +scales
    return total_bits / orig_params


class LowRankGemmaQuantizer:
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

        print(f"Found {len(self.weight_info)} weights")

    def load_tensor(self, key: str, info: dict) -> torch.Tensor:
        begin, end = info['data_offsets']
        numpy_dtype = {
            'F16': np.float16, 'BF16': np.float16, 'F32': np.float32,
        }.get(info['dtype'], np.float32)

        with open(self.safetensor_path, 'rb') as f:
            f.seek(8 + struct.calcsize('Q') + begin)
            data = f.read(end - begin)

        return torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape'])

    def stream_weights(self):
        for idx, (key, info) in enumerate(self.weight_info):
            W = self.load_tensor(key, info).float()
            yield idx, key, W, info['shape']
            del W
            gc.collect()

    def quantize_adaptive_rank(self, target_bpw: float = 0.5, sigma_bits: int = 2) -> Dict:
        """Quantize with rank adapted to achieve target bits per weight"""
        print(f"\nAdaptive Rank Quantization (target_bpw={target_bpw})")

        quantized = {}
        stats = {'total_original': 0, 'total_bits': 0, 'layer_stats': []}

        for idx, key, W, orig_shape in self.stream_weights():
            orig_params = W.numel()
            stats['total_original'] += orig_params

            m, n = W.shape
            max_possible_rank = min(m, n)

            # Binary search or direct compute: find rank that gives target bpw
            # bpw = (m*rank + rank + n*rank) * 1 + rank * sigma_bits) / (m*n)
            #      = rank * (m + n + 1) * 1 + rank * sigma_bits) / (m*n)
            # For ternary U/Vt (1 bit each) and sigma_bits for S:
            # bpw ≈ rank * (m + n + 1) / (m*n) + rank * sigma_bits / (m*n)
            # bpw ≈ rank * ((m + n + 1) + sigma_bits) / (m*n)

            # Solve for rank:
            # target_bpw * m * n = rank * ((m + n + 1) + sigma_bits)
            # rank = target_bpw * m * n / ((m + n + 1) + sigma_bits)

            # But we want to use energy threshold too, so compute both and take min
            # First compute SVD
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            cum_energy = (S ** 2).cumsum(dim=0) / (S ** 2).sum()

            # Try different thresholds to find one that gives target rank
            best_rank = max_possible_rank
            best_bpw = 1.0

            # For bpw=0.5, compute what rank we'd need
            # 0.5 = rank * ((m + n + 1) + sigma_bits) / (m*n)
            # rank = 0.5 * m * n / ((m + n + 1) + sigma_bits)

            target_rank_formula = int(target_bpw * m * n / ((m + n + 1) + sigma_bits))

            # Find energy threshold that gives close to target rank
            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                energy_rank = (cum_energy < thresh).sum().item() + 1
                energy_rank = min(energy_rank, max_possible_rank)

                # Compute actual bpw
                u_bits = m * energy_rank
                s_bits = energy_rank * sigma_bits
                vt_bits = energy_rank * n
                total_bits = u_bits + s_bits + vt_bits + 32  # scales
                actual_bpw = total_bits / orig_params

                if actual_bpw <= target_bpw * 1.2:  # Within 20% of target
                    if energy_rank < best_rank:
                        best_rank = energy_rank
                        best_bpw = actual_bpw

            # Use target rank if it's smaller than what energy gives
            if target_rank_formula < best_rank:
                best_rank = target_rank_formula
                # Recompute bpw
                u_bits = m * best_rank
                s_bits = best_rank * sigma_bits
                vt_bits = best_rank * n
                total_bits = u_bits + s_bits + vt_bits + 32
                best_bpw = total_bits / orig_params

            # Truncate SVD factors
            U_r = U[:, :best_rank]
            S_r = S[:best_rank]
            Vt_r = Vt[:best_rank, :]

            # Quantize
            U_scale = U_r.abs().max()
            if U_scale > 0:
                U_q = (U_r / U_scale).round().clamp(-1, 1).to(torch.int8)
            else:
                U_q = torch.zeros_like(U_r).to(torch.int8)

            Vt_scale = Vt_r.abs().max()
            if Vt_scale > 0:
                Vt_q = (Vt_r / Vt_scale).round().clamp(-1, 1).to(torch.int8)
            else:
                Vt_q = torch.zeros_like(Vt_r).to(torch.int8)

            S_scale = S_r.abs().max()
            qmax = 2 ** (sigma_bits - 1) - 1
            if S_scale > 0:
                scale = S_scale / qmax
                S_q = (S_r / scale).round().clamp(-qmax, qmax).to(torch.int8)
            else:
                S_q = torch.zeros_like(S_r).to(torch.int8)
                scale = 1.0

            # Reconstruct to check quality
            W_recon = torch.matmul(U_q.float() * U_scale * S_q.float().unsqueeze(0) * scale, Vt_q.float())

            quantized[idx] = {
                'U_q': U_q,
                'U_scale': U_scale.item(),
                'Vt_q': Vt_q,
                'Vt_scale': Vt_scale.item(),
                'S_q': S_q,
                'S_scale': scale.item(),
                'rank': best_rank,
                'shape': list(orig_shape),
                'key': key
            }

            stats['total_bits'] += U_q.numel() + Vt_q.numel() + S_q.numel()
            stats['total_bits'] += 32  # scales
            stats['layer_stats'].append({
                'idx': idx,
                'key': key[:40],
                'rank': best_rank,
                'bpw': best_bpw
            })

            if idx % 50 == 0 or idx < 5:
                print(f"  Layer {idx}: rank={best_rank}, bpw={best_bpw:.4f}, shape={list(orig_shape)}")

            del W, U, S, Vt
            gc.collect()

        stats['avg_bpw'] = stats['total_bits'] / stats['total_original']
        stats['compression'] = stats['total_original'] * 16 / stats['total_bits']

        return quantized, stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--output', default='quantized/gemma_lrquant.pt')
    parser.add_argument('--target-bpw', type=float, default=0.5)
    parser.add_argument('--sigma-bits', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    quantizer = LowRankGemmaQuantizer(args.model_dir, args.device)
    quantized, stats = quantizer.quantize_adaptive_rank(
        target_bpw=args.target_bpw,
        sigma_bits=args.sigma_bits
    )

    print(f"\n=== Results ===")
    print(f"Layers: {len(quantized)}")
    print(f"Average bits/weight: {stats['avg_bpw']:.4f}")
    print(f"Compression vs FP16: {stats['compression']:.1f}x")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        'quantized': quantized,
        'stats': stats,
        'method': 'adaptive_rank',
        'config': {
            'target_bpw': args.target_bpw,
            'sigma_bits': args.sigma_bits
        }
    }, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()