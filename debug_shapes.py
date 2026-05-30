import struct

with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    # Skip header
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

    # Skip tensor infos until we find U_shape
    n_tensors = struct.unpack('<Q', f.read(8))[0]

    for i in range(n_tensors):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]

        if name == 'model.layers.0.U_shape':
            print(f'Found at offset {offset}')
            print(f'shape: {shape}, dtype: {dtype}')
            f.seek(tensor_data_start + offset)
            data = f.read(8)
            print(f'Data bytes: {data.hex()}')
            print(f'As little-endian int32[2]: {struct.unpack("<ii", data)}')

    # Now compare with what we wrote
    import torch
    import numpy as np

    data_pt = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
    q = data_pt['quantized']
    e = q[0]

    U_shape = np.array(e['U_shape'], dtype=np.int32)
    print(f'\nOriginal U_shape from PT: {U_shape}')
    print(f'As bytes: {U_shape.tobytes().hex()}')
    print(f'As int32[2]: {struct.unpack("<ii", U_shape.tobytes())}')