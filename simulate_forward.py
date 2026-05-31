"""
Simulated forward pass through gemma-4-E2B-sub1bit.
Run all 316 layers in sequence to measure performance.
"""

import torch
import time
from pathlib import Path
from stream_inference_gguf import GGUFReader, ternary_gemv

def simulate_forward_pass(max_layers=None):
    """Simulate a full forward pass by processing all layers."""
    gguf_path = 'quantized/gemma-4-E2B-sub1bit-stream.gguf'

    print(f"Loading {gguf_path}...")
    print(f"Size: {Path(gguf_path).stat().st_size / 1e6:.1f} MB")

    with GGUFReader(gguf_path) as reader:
        print(f"Metadata: {reader.metadata}")

        # Count total layers
        total_layers = len([k for k in reader.tensors.keys() if k.endswith('.U_packed')])
        print(f"Total layers: {total_layers}")

        if max_layers:
            total_layers = min(max_layers, total_layers)

        # Simulate a sequence of length 1, hidden_dim 1536
        # The input to each "layer" is the output from previous
        # But each layer has different U/Vt dimensions...

        # Actually, let's just measure how long each GEMV takes
        # and report the total time

        print(f"\nProcessing {total_layers} layers...")
        total_time = 0
        layer_times = []

        # For simplicity, use a fixed input size
        # Input dimension varies per layer - we need to track that

        for layer_idx in range(total_layers):
            tensors = reader.get_layer(layer_idx)
            if not tensors:
                break

            U_shape = tuple(tensors['U_shape'].tolist())
            Vt_shape = tuple(tensors['Vt_shape'].tolist())

            # Input must match Vt's output dimension
            x = torch.randn(Vt_shape[1])

            start = time.perf_counter()
            y = ternary_gemv(
                tensors['U_packed'],
                tensors['Vt_packed'],
                tensors['S'],
                tensors['U_scale'].item(),
                tensors['Vt_scale'].item(),
                tensors['S_scale'].item(),
                x,
                U_shape,
                Vt_shape
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
            layer_times.append((layer_idx, elapsed, U_shape, Vt_shape, y.shape))

            if layer_idx % 50 == 0 or layer_idx < 5:
                print(f"Layer {layer_idx}: {elapsed*1000:.2f}ms, U={U_shape}, Vt={Vt_shape}, output={y.shape}")

        avg_time = sum(t for _, t, _, _, _ in layer_times) / len(layer_times)
        print(f"\n=== Results ===")
        print(f"Layers processed: {len(layer_times)}")
        print(f"Average time: {avg_time*1000:.2f}ms per layer")
        print(f"Total time: {total_time:.2f}s")

        # Show timing distribution
        times = [t for _, t, _, _, _ in layer_times]
        times.sort()
        print(f"Min: {min(times)*1000:.2f}ms, Max: {max(times)*1000:.2f}ms, Median: {times[len(times)//2]*1000:.2f}ms")

        # Estimate for different batch sizes
        print(f"\nEstimated times:")
        print(f"  Single token: {total_time:.2f}s")
        print(f"  32 tokens: {total_time*32:.2f}s (batch parallel)")
        print(f"  512 tokens: {total_time*512:.2f}s (full context)")

        return layer_times


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-layers', type=int, default=None)
    args = parser.parse_args()

    simulate_forward_pass(max_layers=args.max_layers)