import struct
import numpy as np

with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_meta = struct.unpack('<Q', f.read(8))[0]

    # skip metadata
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

    # Read tensor info for layer 0
    for i in range(n_tensors):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]

        if '0.Vt_shape' in name:
            print(f'{name}: shape={shape}, dtype={dtype}, offset={offset}')

            # Read data
            f.seek(tensor_data_start + offset)
            data = f.read(8)
            print(f'  Raw bytes: {data.hex()}')
            print(f'  As int32[2]: {struct.unpack("<ii", data)}')

            # Expected
            expected = np.array([1155, 6144], dtype=np.int32)
            print(f'  Expected (1155, 6144): {expected.tobytes().hex()} = {struct.unpack("<ii", expected.tobytes())}')