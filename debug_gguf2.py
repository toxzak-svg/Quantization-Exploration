"""Debug GGUF shape reading - find U_shape tensor."""

import torch
import struct

with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_metadata = struct.unpack('<Q', f.read(8))[0]

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

    tensor_data_start = f.tell()

    # Find U_shape tensor
    name_to_info = {}
    for i in range(n_tensors):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        name_to_info[name] = {'shape': shape, 'dtype': dtype, 'offset': offset}

    # Find and read U_shape for layer 0
    target = 'model.layers.0.U_shape'
    if target in name_to_info:
        info = name_to_info[target]
        print(f"Found {target}:")
        print(f"  shape: {info['shape']}")
        print(f"  dtype: {info['dtype']}")
        print(f"  offset: {info['offset']}")

        # Read data
        f.seek(tensor_data_start + info['offset'])
        data = f.read(8)  # 2 int32 values
        print(f"  Raw bytes: {data.hex()}")
        print(f"  As int32: {struct.unpack('<ii', data)}")
    else:
        print(f"{target} not found!")
        # Find similar
        for name in name_to_info:
            if 'shape' in name.lower():
                print(f"  Found: {name}")