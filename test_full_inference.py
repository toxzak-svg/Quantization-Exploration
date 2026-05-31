"""
Test full inference on gemma-4-E2B-sub1bit GGUF.
Load all layers and run a complete forward pass.
"""

import torch
import time
from stream_inference_gguf import GGUFReader, ternary_gemv, unpack_ternary

def test_full_inference(max_layers=50):
    """Test full forward pass through all/most layers."""
    gguf_path = 'quantized/gemma-4-E2B-sub1bit-stream.gguf'

    print(f"Loading {gguf_path}...")
    print(f"Size: {Path(gguf_path).stat().st_size / 1e6:.1f} MB")

    with GGUFReader(gguf_path) as reader:
        print(f"Metadata: {reader.metadata}")
        print(f"Tensors: {len(reader.tensors)}")

        # Get first layer to establish dimensions
        layer0 = reader.get_layer(0)
        seq_len = 512  # Simulate sequence length

        # Create a fake input (embedding output)
        hidden_dim = 1536  # embedding dimension
        x = torch.randn(seq_len, hidden_dim)

        print(f"\nRunning {max_layers} layers...")
        print(f"Input shape: {x.shape}")

        total_time = 0
        for layer_idx in range(max_layers):
            tensors = reader.get_layer(layer_idx)
            if not tensors:
                break

            U_shape = tuple(tensors['U_shape'].tolist())
            Vt_shape = tuple(tensors['Vt_shape'].tolist())

            start = time.perf_counter()
            y = ternary_gemv(
                tensors['U_packed'],
                tensors['Vt_packed'],
                tensors['S'],
                tensors['U_scale'].item(),
                tensors['Vt_scale'].item(),
                tensors['S_scale'].item(),
                x[:, :Vt_shape[1]],  # Input must match Vt dimension
                U_shape,
                Vt_shape
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed

            if layer_idx < 5 or layer_idx % 10 == 0:
                print(f"Layer {layer_idx}: {elapsed*1000:.1f}ms, U={U_shape}, Vt={Vt_shape}, out={y.shape}")

        avg_time = total_time / max_layers * 1000
        print(f"\n=== Summary ===")
        print(f"Layers processed: {max_layers}")
        print(f"Average time: {avg_time:.1f}ms per layer")
        print(f"Total time: {total_time:.1f}s")

        # Estimate for all 316 layers
        estimated_full = (total_time / max_layers) * 316
        print(f"Estimated full pass: {estimated_full:.1f}s for 316 layers")


if __name__ == '__main__':
    from pathlib import Path
    test_full_inference(max_layers=50)