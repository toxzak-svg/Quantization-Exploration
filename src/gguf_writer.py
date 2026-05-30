import struct
import os
import numpy as np
from typing import List, Tuple, Any


GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3


GGUF_TYPES = {
    'uint8': 0, 'int8': 1,
    'uint16': 2, 'int16': 3,
    'uint32': 4, 'int32': 5,
    'float32': 6, 'bool': 7,
    'string': 8, 'array': 9,
    'uint64': 10, 'int64': 11, 'float64': 12,
}


GGML_TYPES = {
    'float32': 0, 'float16': 1,
    'q8_0': 2, 'q4_0': 3, 'q4_1': 4,
    'int8': 5, 'int16': 6, 'int32': 7,
}


class GGUFWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.kv_pairs: List[Tuple[str, Any]] = []
        self.tensors: List[Tuple[str, np.ndarray, int]] = []

    def add_key_value(self, key: str, value: Any):
        self.kv_pairs.append((key, value))

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int = 0):
        self.tensors.append((name, data, ggml_type))

    def write(self):
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, 'wb') as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.kv_pairs)))
            for key, value in self.kv_pairs:
                self._write_string(f, key)
                self._write_value(f, value)
            sizes = [data.nbytes for _, data, _ in self.tensors]
            data_offset = f.tell() + sum(
                8 + len(name.encode('utf-8')) + 4 + 8 * data.ndim + 4 + 8
                for name, data, _ in self.tensors
            )
            for i, (name, data, ggml_type) in enumerate(self.tensors):
                self._write_string(f, name)
                f.write(struct.pack('<I', data.ndim))
                for dim in data.shape:
                    f.write(struct.pack('<q', dim))
                f.write(struct.pack('<I', ggml_type))
                f.write(struct.pack('<Q', data_offset))
                data_offset += sizes[i]
            for _, data, _ in self.tensors:
                f.write(data.tobytes())

    def _write_string(self, f, s: str):
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_value(self, f, value: Any):
        if isinstance(value, str):
            f.write(struct.pack('<i', GGUF_TYPES['string']))
            self._write_string(f, value)
        elif isinstance(value, bool):
            f.write(struct.pack('<i', GGUF_TYPES['bool']))
            f.write(struct.pack('<b', 1 if value else 0))
        elif isinstance(value, int):
            f.write(struct.pack('<i', GGUF_TYPES['int32']))
            f.write(struct.pack('<i', value))
        elif isinstance(value, float):
            f.write(struct.pack('<i', GGUF_TYPES['float32']))
            f.write(struct.pack('<f', value))
        elif isinstance(value, list):
            f.write(struct.pack('<i', GGUF_TYPES['array']))
            if value:
                f.write(struct.pack('<i', GGUF_TYPES.get(type(value[0]).__name__, 8)))
            f.write(struct.pack('<Q', len(value)))
            for item in value:
                if isinstance(item, str):
                    self._write_string(f, item)
                else:
                    self._write_value(f, item)
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
