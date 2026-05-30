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
print(f'As little-endian int32s: {struct.unpack("<ii", U_shape_np.tobytes())}')

# Now read directly from GGUF at the stated offset
with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    # Skip to tensor data area
    f.read(4 + 4 + 8 + 8)  # magic, version, n_tensors, n_metadata

    # Skip metadata
    n_metadata = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n_metadata):
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
    print(f'\nTensor data starts at: {tensor_data_start}')

    # Skip tensor info for layer 0 (8 tensors)
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    print(f'Total tensors: {n_tensors}')

    # Read all tensor infos and find U_shape offset
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

            if dtype == 5:  # uint8
                values = np.frombuffer(data, dtype=np.uint8)
                print(f'  As uint8[8]: {values}')