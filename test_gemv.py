"""
Simple test: verify GEMV produces reasonable output.
"""

import torch
from stream_inference_gguf import GGUFReader, ternary_gemv

def test_gemv():
    gguf_path = 'quantized/gemma-4-E2B-sub1bit-stream.gguf'

    with GGUFReader(gguf_path) as reader:
        layer0 = reader.get_layer(0)

        U_shape = tuple(layer0['U_shape'].tolist())
        Vt_shape = tuple(layer0['Vt_shape'].tolist())

        print(f"Layer 0:")
        print(f"  U_shape: {U_shape}")
        print(f"  Vt_shape: {Vt_shape}")
        print(f"  S_shape: {layer0['S'].shape}")
        print(f"  U_scale: {layer0['U_scale'].item():.4f}")
        print(f"  Vt_scale: {layer0['Vt_scale'].item():.4f}")
        print(f"  S_scale: {layer0['S_scale'].item():.4f}")

        # GEMV: y = (U @ diag(S) @ Vt) @ x
        # Input x must be [Vt_shape[1]] = [6144]
        # Output y will be [U_shape[0]] = [1536]

        x = torch.randn(Vt_shape[1])
        print(f"\nInput x shape: {x.shape}")

        y = ternary_gemv(
            layer0['U_packed'],
            layer0['Vt_packed'],
            layer0['S'],
            layer0['U_scale'].item(),
            layer0['Vt_scale'].item(),
            layer0['S_scale'].item(),
            x,
            U_shape,
            Vt_shape
        )

        print(f"Output y shape: {y.shape}")
        print(f"y mean: {y.mean():.4f}, std: {y.std():.4f}, abs_max: {y.abs().max():.4f}")

        # Check how this compares to just using Vt @ x (partial computation)
        Vt = torch.zeros(Vt_shape, dtype=torch.float32)
        for i in range(Vt_shape[0]):
            for j in range(Vt_shape[1]):
                pass  # Skip, we need unpack

        print("\nTesting multiple layers...")
        for layer_idx in [0, 1, 10, 50, 100]:
            tensors = reader.get_layer(layer_idx)
            if not tensors:
                continue

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

            print(f"Layer {layer_idx}: U={U_shape}, Vt={Vt_shape}, y.shape={y.shape}, mean={y.mean():.3f}, std={y.std():.3f}")


if __name__ == '__main__':
    test_gemv()