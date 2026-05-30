"""Debug exactly what's at U_shape offset."""

import struct

with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    # Read header
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_metadata = struct.unpack('<Q', f.read(8))[0]

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
    print(f'Tensor data starts at: {tensor_data_start}')

    # Read all tensor info to build offset table
    tensor_infos = {}
    for i in range(n_tensors):
        name_len = struct.unpack('<I', f.read(4))[0]
        name = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensor_infos[name] = {
            'shape': shape,
            'dtype': dtype,
            'offset': offset
        }

    # Check layer 0 tensors
    for suffix in ['U_packed', 'Vt_packed', 'S', 'U_scale', 'Vt_scale', 'U_shape', 'Vt_shape']:
        name = f'model.layers.0.{suffix}'
        if name in tensor_infos:
            info = tensor_infos[name]
            actual_offset = tensor_data_start + info['offset']
            print(f'\n{name}:')
            print(f'  shape={info["shape"]}, dtype={info["dtype"]}, file_offset={info["offset"]}')
            print(f'  Absolute position: {actual_offset}')

            # Read data at this offset
            f.seek(actual_offset)
            if info['dtype'] == 4:  # int32
                data = f.read(8)
                print(f'  Data (hex): {data.hex()}')
                values = struct.unpack('<ii', data)
                print(f'  As int32[2]: {values}')
            elif info['dtype'] == 2:  # float32
                data = f.read(4)
                print(f'  Data (hex): {data.hex()}')
                value = struct.unpack('<f', data)[0]
                print(f'  As float32: {value}')
            elif info['dtype'] == 5:  # uint8
                data = f.read(min(16, info['shape'][0]))
                print(f'  First {len(data)} bytes (hex): {data.hex()}')
            elif info['dtype'] == 6:  # int8
                data = f.read(min(16, info['shape'][0]))
                print(f'  First {len(data)} bytes (hex): {data.hex()}')

    # Now verify by computing expected offset
    print('\n--- Verification ---')
    # U_packed: offset 326, size 354816
    # Vt_packed: offset 355142, size 1419264
    # S: offset 1774406, size 1155
    # U_scale: offset 1775561, size 4
    # Vt_scale: offset 1775565, size 4
    # U_shape: offset 1775569, size 8

    expected_U_shape_offset = 326 + 354816 + 1419264 + 1155 + 4 + 4
    print(f'Expected U_shape offset (from sequential calc): {expected_U_shape_offset}')
    print(f'Actual U_shape offset from tensor info: {tensor_infos["model.layers.0.U_shape"]["offset"]}')

    # Check if they match
    if expected_U_shape_offset != tensor_infos["model.layers.0.U_shape"]["offset"]:
        print('MISMATCH! Tensor offsets are not sequential!')
    else:
        print('Sequential offset is correct')

    # Now read at expected offset
    f.seek(tensor_data_start + expected_U_shape_offset)
    data = f.read(8)
    print(f'\nData at expected offset: {data.hex()}')
    values = struct.unpack('<ii', data)
    print(f'As int32[2]: {values}')