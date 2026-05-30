import struct

# Read GGUF properly
with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    # Read header
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_meta = struct.unpack('<Q', f.read(8))[0]

    # Read metadata
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

    after_meta = f.tell()
    print(f'After metadata: {after_meta}')

    # Read ALL tensor infos first
    tensor_infos = []
    for i in range(n_tensors):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensor_infos.append((name, shape, dtype, offset))

    data_start = f.tell()
    print(f'Data starts at: {data_start}')

    # Find layer 0 tensors
    print('\nLayer 0 tensors:')
    for name, shape, dtype, offset in tensor_infos[:8]:
        actual_pos = data_start + offset
        f.seek(actual_pos)

        if 'scale' in name.lower():
            data = f.read(4)
            val = struct.unpack('<f', data)[0]
            print(f'  {name}: offset={offset}, actual={actual_pos}, value={val:.6f}')
        elif 'shape' in name.lower():
            data = f.read(8)
            vals = struct.unpack('<ii', data)
            print(f'  {name}: offset={offset}, actual={actual_pos}, value={vals}')
        else:
            data = f.read(4)
            print(f'  {name}: offset={offset}, actual={actual_pos}, first_bytes={data.hex()}')

import torch
import numpy as np
data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
q = data['quantized']
e = q[0]
print()
print(f"Expected U_scale: {e['U_scale'].item():.6f}")
print(f"Expected Vt_scale: {e['Vt_scale'].item():.6f}")
print(f"Expected S_scale: {e['S_scale'].item():.6f}")
print(f"Expected U_shape: {e['U_shape']}")
print(f"Expected Vt_shape: {e['Vt_shape']}")