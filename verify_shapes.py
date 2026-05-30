import struct
import numpy as np

# Read Vt_shape from freshly built GGUF
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

    tensor_infos = []
    for i in range(n_tensors):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensor_infos.append((name, shape, dtype, offset))

    for name, shape, dtype, offset in tensor_infos:
        if name == 'model.layers.0.Vt_shape':
            print(f'Vt_shape: offset={offset}')
            f.seek(tensor_data_start + offset)
            data = f.read(8)
            print(f'Raw bytes: {data.hex()}')
            values = struct.unpack('<ii', data)
            print(f'As int32[2]: {values}')

            expected = np.array([1155, 6144], dtype=np.int32)
            expected_bytes = expected.tobytes()
            print(f'Expected bytes: {expected_bytes.hex()}')
            expected_values = struct.unpack('<ii', expected_bytes)
            print(f'Expected values: {expected_values}')

            match = values == expected_values
            print(f'Match: {match}')
        elif name == 'model.layers.0.U_shape':
            print(f'U_shape: offset={offset}')
            f.seek(tensor_data_start + offset)
            data = f.read(8)
            print(f'Raw bytes: {data.hex()}')
            values = struct.unpack('<ii', data)
            print(f'As int32[2]: {values}')

            expected = np.array([1536, 1155], dtype=np.int32)
            expected_bytes = expected.tobytes()
            print(f'Expected bytes: {expected_bytes.hex()}')
            expected_values = struct.unpack('<ii', expected_bytes)
            print(f'Expected values: {expected_values}')

            match = values == expected_values
            print(f'Match: {match}')