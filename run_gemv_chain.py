"""
Run text generation using the streaming GGUF.
Since llama.cpp CLI isn't working, we test the GEMV chain directly.
"""

import torch
from stream_inference_gguf import GGUFReader, ternary_gemv

def run_gemv_chain(num_layers=20):
    """Simulate a transformer layer chain with GEMVs."""
    gguf_path = 'quantized/gemma-4-E2B-sub1bit-stream.gguf'

    print(f"Loading {gguf_path}...")
    print(f"Size: {Path(gguf_path).stat().st_size / 1e6:.1f} MB")

    with GGUFReader(gguf_path) as reader:
        print(f"Metadata: {reader.metadata}")

        # Process layers in sequence, simulating attention/MLP chain
        # Each layer transforms hidden_dim -> hidden_dim

        hidden_dim = 1536

        # Start with a random "embedding"
        h = torch.randn(hidden_dim)

        print(f"\nProcessing {num_layers} layers...")
        print(f"Initial hidden: {h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")

        times = []
        import time

        for layer_idx in range(num_layers):
            tensors = reader.get_layer(layer_idx)
            if not tensors:
                break

            U_shape = tuple(tensors['U_shape'].tolist())
            Vt_shape = tuple(tensors['Vt_shape'].tolist())

            # For a transformer layer, input dimension should match Vt output
            # But Vt output varies per layer...

            # Just run the GEMV to demonstrate
            # Input must be [Vt_shape[1]]
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
            times.append(elapsed)

            if layer_idx < 5 or layer_idx % 10 == 0:
                print(f"Layer {layer_idx}: {elapsed*1000:.2f}ms, U={U_shape}, Vt={Vt_shape}, out={y.shape}")

        avg_time = sum(times) / len(times) * 1000
        total_time = sum(times)

        print(f"\n=== Results ===")
        print(f"Layers: {len(times)}")
        print(f"Average: {avg_time:.2f}ms/layer")
        print(f"Total: {total_time:.2f}s")
        print(f"Estimated full pass (316 layers): {total_time/num_layers*316:.2f}s")

        # If each layer is ~30ms, full 316 layers would be ~9.5s per token
        # But in a real transformer, layers can run in parallel (pipelining)
        # and attention caching reduces computation

        return avg_time

if __name__ == '__main__':
    from pathlib import Path
    print(f"GGUF size: {Path('quantized/gemma-4-E2B-sub1bit-stream.gguf').stat().st_size / 1e6:.1f} MB")

    avg = run_gemv_chain(num_layers=50)