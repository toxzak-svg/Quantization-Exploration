import torch
import struct
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3


gguf_type_map = {
    'uint8': 0,
    'int8': 1,
    'uint16': 2,
    'int16': 3,
    'uint32': 4,
    'int32': 5,
    'float32': 6,
    'bool': 7,
    'string': 8,
    'array': 9,
    'uint64': 10,
    'int64': 11,
    'float64': 12,
}


@dataclass
class QuantizedLayer:
    U: torch.Tensor
    S: torch.Tensor
    Vt: torch.Tensor
    U_scale: torch.Tensor
    S_scale: torch.Tensor
    bit_allocations: Tuple[int, int, int]


class GGUFWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.kv_pairs: List[Tuple[str, any]] = []
        self.tensors: List[Tuple[str, np.ndarray, str]] = []

    def add_key_value(self, key: str, value: any):
        self.kv_pairs.append((key, value))

    def add_tensor(self, name: str, data: np.ndarray, tensor_type: str = "float32"):
        self.tensors.append((name, data, tensor_type))

    def write(self):
        with open(self.filepath, 'wb') as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack('<I', GGUF_VERSION))

            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.kv_pairs)))

            for key, value in self.kv_pairs:
                self._write_string(f, key)
                self._write_value(f, value)

            data_offset = f.tell()
            tensor_data_sizes = []
            for name, data, _ in self.tensors:
                tensor_data_sizes.append(data.nbytes)

            for i, (name, data, tensor_type) in enumerate(self.tensors):
                self._write_string(f, name)
                f.write(struct.pack('<I', data.ndim))
                for dim in data.shape:
                    f.write(struct.pack('<q', dim))
                f.write(struct.pack('<I', ggml_type_to_id(tensor_type)))
                f.write(struct.pack('<Q', data_offset))
                data_offset += tensor_data_sizes[i]

            for _, data, _ in self.tensors:
                f.write(data.tobytes())

    def _write_string(self, f, s: str):
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_value(self, f, value: any):
        if isinstance(value, str):
            f.write(struct.pack('<i', gguf_type_map['string']))
            self._write_string(f, value)
        elif isinstance(value, bool):
            f.write(struct.pack('<i', gguf_type_map['bool']))
            f.write(struct.pack('<b', 1 if value else 0))
        elif isinstance(value, int):
            f.write(struct.pack('<i', gguf_type_map['int32']))
            f.write(struct.pack('<i', value))
        elif isinstance(value, float):
            f.write(struct.pack('<i', gguf_type_map['float32']))
            f.write(struct.pack('<f', value))
        elif isinstance(value, list):
            f.write(struct.pack('<i', gguf_type_map['array']))
            f.write(struct.pack('<i', gguf_type_map['string']))
            f.write(struct.pack('<Q', len(value)))
            for item in value:
                self._write_string(f, item)
        else:
            raise ValueError(f"Unsupported type: {type(value)}")


def ggml_type_to_id(type_name: str) -> int:
    type_map = {
        'float32': 0,
        'float16': 1,
        'quat8': 2,
        'quat4': 3,
        'quart2': 4,
        'int8': 5,
        'int16': 6,
        'int32': 7,
    }
    return type_map.get(type_name, 0)


def pack_sub1bit_model(
    factors: Dict[int, Dict],
    output_path: str,
    model_name: str = "llama-2-7b-sub1bit",
    metadata: Optional[Dict] = None
):
    writer = GGUFWriter(output_path)

    writer.add_key_value("general.architecture", "llama")
    writer.add_key_value("general.name", model_name)
    writer.add_key_value("general.file_type", "sub1bit")

    if metadata:
        for key, value in metadata.items():
            writer.add_key_value(key, value)

    for layer_idx, layer_data in factors.items():
        U = layer_data['U'].cpu().numpy().astype(np.float16)
        S = layer_data['S'].cpu().numpy().astype(np.float16)
        Vt = layer_data['Vt'].cpu().numpy().astype(np.float16)

        writer.add_tensor(f"layer.{layer_idx}.U", U, "float16")
        writer.add_tensor(f"layer.{layer_idx}.S", S, "float16")
        writer.add_tensor(f"layer.{layer_idx}.Vt", Vt, "float16")

    writer.write()

    file_size = os.path.getsize(output_path)
    return file_size


def create_lowrank_type_defs() -> str:
    return """
// Sub-1-bit quantization type definitions
// GGML_TYPE_LOWRANK_UV_0BIT  - U/V factors skipped (rank 0)
// GGML_TYPE_LOWRANK_UV_1BIT  - U/V quantized to ternary {0, +-1}
// GGML_TYPE_LOWRANK_SIGMA_2BIT - singular values quantized to 2-bit

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q8_0    = 2,
    GGML_TYPE_Q4_0    = 3,
    GGML_TYPE_Q4_1    = 4,
    // ... existing types ...
    GGML_TYPE_LOWRANK_UV_0BIT  = 100,
    GGML_TYPE_LOWRANK_UV_1BIT  = 101,
    GGML_TYPE_LOWRANK_SIGMA_2BIT = 102,
};
"""


if __name__ == "__main__":
    dummy_factors = {
        0: {
            'U': torch.randn(4096, 16),
            'S': torch.randn(16),
            'Vt': torch.randn(16, 4096),
            'rank': 16
        }
    }

    output_path = "C:/Users/Zwmar/projects/sub1quant/quantized/test.gguf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    size = pack_sub1bit_model(dummy_factors, output_path)
    print(f"GGUF file created: {size} bytes")