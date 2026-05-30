"""
Streaming inference from sub1bit GGUF (<1GB).
Fixed: offsets are relative to tensor data start.
"""

import torch
import struct
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np


class GGUFReader:
    """Efficient GGUF reader with proper offset handling."""

    def __init__(self, path: str):
        self.path = path
        self.metadata = {}
        self.tensors = {}
        self._f = None
        self._data_start = 0

    def open(self):
        self._f = open(self.path, 'rb')

        magic = struct.unpack('<I', self._f.read(4))[0]
        assert magic == 0x46554747, "Not a GGUF file"

        version = struct.unpack('<I', self._f.read(4))[0]
        n_tensors = struct.unpack('<Q', self._f.read(8))[0]
        n_metadata = struct.unpack('<Q', self._f.read(8))[0]

        # Read metadata
        for _ in range(n_metadata):
            key_len = struct.unpack('<I', self._f.read(4))[0]
            key = self._f.read(key_len).decode('utf-8')
            val_type = struct.unpack('<I', self._f.read(4))[0]
            if val_type == 8:
                val_len = struct.unpack('<Q', self._f.read(8))[0]
                val = self._f.read(val_len).decode('utf-8')
            elif val_type == 9:
                val = struct.unpack('<f', self._f.read(4))[0]
            elif val_type == 4:
                val = struct.unpack('<i', self._f.read(4))[0]
            self.metadata[key] = val

        # After metadata, we're at tensor info
        # Tensor data starts after all tensor info
        tensor_info_start = self._f.tell()

        # Read tensor info
        for i in range(n_tensors):
            name_len = struct.unpack('<I', self._f.read(4))[0]
            name = self._f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<Q', self._f.read(8))[0]
            shape = [struct.unpack('<Q', self._f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', self._f.read(4))[0]
            offset = struct.unpack('<Q', self._f.read(8))[0]
            self.tensors[name] = {
                'shape': shape,
                'dtype': dtype,
                'offset': offset,
            }

        self._data_start = self._f.tell()

        return self

    def close(self):
        if self._f:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def get(self, name: str):
        if name not in self.tensors:
            return None

        info = self.tensors[name]
        shape = info['shape']
        dtype = info['dtype']
        offset = info['offset']

        dtype_map = {
            0: np.float32, 1: np.float16, 2: np.float32,
            4: np.int32, 5: np.uint8, 6: np.int8, 7: np.uint16, 8: np.int16,
        }

        numpy_dtype = dtype_map.get(dtype, np.float32)
        n_elements = 1
        for d in shape:
            n_elements *= d

        # Use _data_start to get absolute position
        self._f.seek(self._data_start + offset)
        data = self._f.read(n_elements * np.dtype(numpy_dtype).itemsize)
        return torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy().reshape(shape))

    def get_layer(self, layer_idx: int) -> Optional[Dict]:
        tensors = {}
        for suffix in ['U_packed', 'Vt_packed', 'S', 'U_scale', 'Vt_scale', 'S_scale', 'U_shape', 'Vt_shape']:
            name = f'model.layers.{layer_idx}.{suffix}'
            t = self.get(name)
            if t is not None:
                tensors[suffix] = t
        return tensors if tensors else None


def unpack_ternary(packed, shape):
    """Unpack 5 ternary values per byte."""
    out_features, in_features = shape
    total = in_features * out_features

    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=packed.device)
    packed_int = packed.to(torch.int32).unsqueeze(-1)
    expanded = packed_int // weights % 3
    flat = (expanded - 1).flatten()[:total]
    return flat.reshape(shape).to(torch.int8)


def ternary_gemv(
    U_packed, Vt_packed, S,
    U_scale, Vt_scale, S_scale,
    x, U_shape, Vt_shape
):
    """Compute y = (U @ diag(S) @ Vt) @ x."""
    U = unpack_ternary(U_packed, U_shape).float() * U_scale
    Vt = unpack_ternary(Vt_packed, Vt_shape).float() * Vt_scale
    S_float = S.float() * S_scale

    temp1 = torch.matmul(Vt, x)
    temp2 = temp1 * S_float
    y = torch.matmul(U, temp2)
    return y


def test_streaming(num_layers=5):
    """Test streaming inference."""
    gguf_path = 'quantized/gemma-4-E2B-sub1bit-stream.gguf'
    print(f"Opening {gguf_path}...")
    print(f"Size: {Path(gguf_path).stat().st_size / 1e6:.1f} MB")

    with GGUFReader(gguf_path) as reader:
        print(f"Metadata: {reader.metadata}")
        print(f"Tensors: {len(reader.tensors)}")
        print(f"Data starts at: {reader._data_start}")

        # Test reading shapes
        layer0 = reader.get_layer(0)
        if layer0:
            print(f"\nLayer 0 shapes:")
            print(f"  U_shape: {layer0['U_shape'].tolist()}")
            print(f"  Vt_shape: {layer0['Vt_shape'].tolist()}")
            print(f"  U_scale: {layer0['U_scale'].item():.4f}")
            print(f"  Vt_scale: {layer0['Vt_scale'].item():.4f}")
            print(f"  S_scale: {layer0['S_scale'].item():.4f}")

        import time
        total_time = 0

        for layer_idx in range(num_layers):
            tensors = reader.get_layer(layer_idx)
            if not tensors:
                print(f"Layer {layer_idx}: not found")
                continue

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
            total_time += elapsed

            print(f"Layer {layer_idx}: {elapsed*1000:.1f}ms, U={U_shape}, Vt={Vt_shape}, y.shape={y.shape}")

        print(f"\nAverage: {total_time/num_layers*1000:.1f}ms per layer")


if __name__ == '__main__':
    test_streaming()