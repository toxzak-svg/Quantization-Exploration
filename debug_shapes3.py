import struct
import torch
import numpy as np

# First check what we wrote for layer 0 U_shape
pt_data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
q = pt_data['quantized']
e = q[0]

U_shape_np = np.array(e['U_shape'], dtype=np.int32)
print(f'From PT: U_shape = {U_shape_np}')
print(f'As bytes (hex): {U_shape_np.tobytes().hex()}')

# Now read from GGUF
with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    # Read header
    magic = struct.unpack('<I', f.read(4))[0]
    print(f'\nMagic: {hex(magic)}')

    version = struct.unpack('<I', f.read(4))[0]
    print(f'Version: {version}')

    n_tensors = struct.unpack('<Q', f.read(8))[0]
    print(f'n_tensors: {n_tensors}')

    n_metadata = struct.unpack('<Q', f.read(8))[0]
    print(f'n_metadata: {n_metadata}')

    # Read metadata
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
    print(f'\nTensor data starts at: {tensor_data_start}')

    # Read tensor infos
    tensor_infos = []
    for i in range(n_tensors):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensor_infos.append((name, shape, dtype, offset))

    # Find U_shape for layer 0
    for name, shape, dtype, offset in tensor_infos:
        if '0.U_shape' in name:
            print(f'\n{name}:')
            print(f'  shape={shape}, dtype={dtype}, offset={offset}')

            f.seek(tensor_data_start + offset)
            data = f.read(8)
            print(f'  Raw bytes: {data.hex()}')

            if dtype == 4:  # int32
                values = struct.unpack('<ii', data)
                print(f'  As int32[2]: {values}')

    # Also check Vt_shape
    for name, shape, dtype, offset in tensor_infos:
        if '0.Vt_shape' in name:
            print(f'\n{name}:')
            print(f'  shape={shape}, dtype={dtype}, offset={offset}')

            f.seek(tensor_data_start + offset)
            data = f.read(8)
            print(f'  Raw bytes: {data.hex()}')

            if dtype == 4:  # int32
                values = struct.unpack('<ii', data)
                print(f'  As int32[2]: {values}')