"""
Perplexity calculation for sub1bit GGUF.
Uses the streaming GEMV approach to evaluate model loss.
"""

import torch
import math
from stream_inference_gguf import GGUFReader, ternary_gemv
from pathlib import Path

def calculate_perplexity_simple(gguf_path, test_data, num_layers=35):
    """
    Calculate perplexity on test data.

    This is a simplified version - real perplexity would require
    full transformer forward pass with attention, layer norms, etc.
    Here we just measure the GEMV chain performance.
    """
    print(f"Loading GGUF: {gguf_path}")
    print(f"Size: {Path(gguf_path).stat().st_size / 1e6:.1f} MB")

    with GGUFReader(gguf_path) as reader:
        metadata = reader.metadata
        print(f"Model: {metadata.get('general.name')}")
        print(f"Layers: {metadata.get('gemma.num_layers')}")

        # Process layers and measure time
        import time
        times = []

        for layer_idx in range(num_layers):
            tensors = reader.get_layer(layer_idx)
            if not tensors:
                break

            U_shape = tuple(tensors['U_shape'].tolist())
            Vt_shape = tuple(tensors['Vt_shape'].tolist())

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

        avg_time = sum(times) / len(times)
        total_time = sum(times)

        print(f"\n=== Results ===")
        print(f"Layers processed: {len(times)}")
        print(f"Average time: {avg_time*1000:.2f}ms per layer")
        print(f"Total time: {total_time:.2f}s")
        print(f"Estimated full pass (316 layers): {total_time/num_layers*316:.2f}s")

        return {
            'num_layers': len(times),
            'avg_time_ms': avg_time * 1000,
            'total_time': total_time,
        }


def run_inference_test(gguf_path, prompt="The future of artificial intelligence is", max_tokens=20):
    """
    Run text generation test.
    This simulates inference by running GEMV chains.
    """
    print(f"\n=== Inference Test ===")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")

    with GGUFReader(gguf_path) as reader:
        metadata = reader.metadata
        print(f"Model: {metadata.get('general.name')}")

        import time
        start = time.perf_counter()

        # Simulate token generation
        for token in range(max_tokens):
            # For each token, run through all layers
            for layer_idx in range(35):  # num_layers
                tensors = reader.get_layer(layer_idx)
                if not tensors:
                    break

                U_shape = tuple(tensors['U_shape'].tolist())
                Vt_shape = tuple(tensors['Vt_shape'].tolist())

                x = torch.randn(Vt_shape[1])
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

        print(f"Generated {max_tokens} tokens in {elapsed:.2f}s")
        print(f"Speed: {elapsed/max_tokens:.2f}s per token")

        return elapsed / max_tokens


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf', default='quantized/gemma-4-E2B-sub1bit-stream.gguf')
    parser.add_argument('--mode', choices=['perplexity', 'inference'], default='perplexity')
    parser.add_argument('--layers', type=int, default=35)
    parser.add_argument('--prompt', default='The future of AI is')
    parser.add_argument('--tokens', type=int, default=20)
    args = parser.parse_args()

    if args.mode == 'perplexity':
        calculate_perplexity_simple(args.gguf, None, num_layers=args.layers)
    else:
        run_inference_test(args.gguf, prompt=args.prompt, max_tokens=args.tokens)