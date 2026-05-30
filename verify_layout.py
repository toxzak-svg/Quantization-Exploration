"""Verify GGUF tensor data layout."""

import torch
import numpy as np

# Load original to compare
pt_data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
q = pt_data['quantized']
e = q[0]

print("Expected values from PT:")
print(f"  U_shape: {e['U_shape']}")
print(f"  Vt_shape: {e['Vt_shape']}")

# Read GGUF and verify
import struct

with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_meta = struct.unpack('<Q', f.read(8))[0]

    for _ in range(n_meta):
        key_len = struct.unpack('<I', f.read(4))[0]
        f.read(key_len)
        val_type = struct.unpack('<I', f.read(4))[0]
        if val_type == 8:
            val_len = struct.unpack('<Q', f.read(8))[0]
            f.read(val_len)
        elif val_type == 9:
            f.read(4)
        elif val_type == 4:
            f.read(4)

    tensor_data_start = f.tell()
    print(f"\nTensor data starts at: {tensor_data_start}")

    # Build tensor info dict
    tensor_infos = {}
    for i in range(n_tensors):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensor_infos[name] = (shape, dtype, offset)

    # Calculate expected offsets for layer 0
    print("\nExpected layout for layer 0:")
    U_packed_size = e['U_packed'].numel()
    Vt_packed_size = e['Vt_packed'].numel()
    S_size = e['S'].numel()
    print(f"  U_packed: size={U_packed_size}")
    print(f"  Vt_packed: size={Vt_packed_size}")
    print(f"  S: size={S_size}")

    cumulative = 0
    expected_offsets = {
        'model.layers.0.U_packed': cumulative,
        'model.layers.0.Vt_packed': cumulative + U_packed_size,
        'model.layers.0.S': cumulative + U_packed_size + Vt_packed_size,
        'model.layers.0.U_scale': cumulative + U_packed_size + Vt_packed_size + S_size,
        'model.layers.0.Vt_scale': cumulative + U_packed_size + Vt_packed_size + S_size + 4,
        'model.layers.0.S_scale': cumulative + U_packed_size + Vt_packed_size + S_size + 8,
        'model.layers.0.U_shape': cumulative + U_packed_size + Vt_packed_size + S_size + 12,
        'model.layers.0.Vt_shape': cumulative + U_packed_size + Vt_packed_size + S_size + 20,
    }

    print(f"\nExpected offsets (relative to tensor_data_start={tensor_data_start}):")
    for name, offset in expected_offsets.items():
        print(f"  {name}: {offset}")

    # Check actual values
    print("\nActual tensor info offsets:")
    for name in expected_offsets.keys():
        if name in tensor_infos:
            shape, dtype, offset = tensor_infos[name]
            print(f"  {name}: offset={offset}")

    # Check actual data at expected offsets
    print("\nData at expected offsets:")
    for name, expected_offset in expected_offsets.items():
        f.seek(tensor_data_start + expected_offset)
        if 'shape' in name.lower():
            data = f.read(8)
            values = struct.unpack('<ii', data)
            print(f"  {name} at {expected_offset}: {values}")
        elif 'scale' in name.lower():
            data = f.read(4)
            value = struct.unpack('<f', data)[0]
            print(f"  {name} at {expected_offset}: {value}")

    # Check actual data at stored offsets
    print("\nData at stored offsets:")
    for name in expected_offsets.keys():
        if name in tensor_infos:
            shape, dtype, stored_offset = tensor_infos[name]
            f.seek(tensor_data_start + stored_offset)
            if 'shape' in name.lower():
                data = f.read(8)
                values = struct.unpack('<ii', data)
                print(f"  {name} at {stored_offset}: {values}")
            elif 'scale' in name.lower():
                data = f.read(4)
                value = struct.unpack('<f', data)[0]
                print(f"  {name} at {stored_offset}: {value}")