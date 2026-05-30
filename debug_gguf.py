"""Debug GGUF shape reading."""

import torch
import struct

# Open the file
with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    # Magic
    magic = struct.unpack('<I', f.read(4))[0]
    print(f"Magic: {hex(magic)}")

    # Version
    version = struct.unpack('<I', f.read(4))[0]
    print(f"Version: {version}")

    # Tensors
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    print(f"Num tensors: {n_tensors}")

    # Metadata
    n_metadata = struct.unpack('<Q', f.read(8))[0]
    print(f"Num metadata: {n_metadata}")

    # Skip metadata
    for _ in range(n_metadata):
        key_len = struct.unpack('<I', f.read(4))[0]
        key = f.read(key_len).decode('utf-8')
        val_type = struct.unpack('<I', f.read(4))[0]
        if val_type == 8:
            val_len = struct.unpack('<Q', f.read(8))[0]
            val = f.read(val_len).decode('utf-8')
        elif val_type == 9:
            val = struct.unpack('<f', f.read(4))[0]
        elif val_type == 4:
            val = struct.unpack('<i', f.read(4))[0]
        print(f"  {key}: {val}")

    tensor_data_start = f.tell()
    print(f"\nTensor data starts at: {tensor_data_start}")

    # Read first few tensor infos
    for i in range(5):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        print(f"\nTensor {i}: {name}")
        print(f"  shape: {shape}, dtype: {dtype}, offset: {offset}")

        # Read the actual data to verify
        if 'shape' in name.lower():
            f.seek(tensor_data_start + offset)
            data = f.read(2 * 4)  # 2 int32 values
            print(f"  Raw data (hex): {data.hex()}")
            print(f"  As int32: {struct.unpack('<ii', data)}")