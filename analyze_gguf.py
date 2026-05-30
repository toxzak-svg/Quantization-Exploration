import struct

path = r'C:\Users\Zwmar\projects\sub1quant\llama-2-7b-sub1bit.gguf'
with open(path, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    print(f'Magic: {magic}, Version: {version}')
    print(f'Tensors: {n_tensors}, KV pairs: {n_kv}')
    
    for i in range(n_kv):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8')
        val_type = struct.unpack('<i', f.read(4))[0]
        if val_type == 8:
            s_len = struct.unpack('<Q', f.read(8))[0]
            val = f.read(s_len).decode('utf-8')
            print(f'  KV[{i}]: {key} = {val}')
        elif val_type == 5:
            val = struct.unpack('<i', f.read(4))[0]
            print(f'  KV[{i}]: {key} = {val}')
        elif val_type == 6:
            val = struct.unpack('<f', f.read(4))[0]
            print(f'  KV[{i}]: {key} = {val}')

    all_tensors = []
    for i in range(n_tensors):
        try:
            name_len = struct.unpack('<Q', f.read(8))[0]
            if name_len > 1000:
                print(f'ERROR at tensor {i}: name_len={name_len}')
                break
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack('<' + 'q'*n_dims, f.read(8*n_dims))
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            all_tensors.append((name, n_dims, dims, dtype, offset))
        except Exception as e:
            print(f'Error at tensor {i}: {e}')
            break

    print(f'\nSuccessfully read {len(all_tensors)} tensors')
    
    if all_tensors:
        dtype_counts = {}
        for name, nd, dims, dt, off in all_tensors:
            dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        print(f'Dtype distribution: {dtype_counts}')
        
        tensor_categories = {}
        for name, nd, dims, dt, off in all_tensors:
            parts = name.split('.')
            suffix = parts[-1]
            if suffix not in tensor_categories:
                tensor_categories[suffix] = []
            tensor_categories[suffix].append((name, dims, dt))
        
        print(f'\nTensor categories:')
        for cat, items in sorted(tensor_categories.items()):
            print(f'  {cat}: {len(items)} tensors, sample shape={items[0][1]}, dtype={items[0][2]}')
            if len(items) < 5:
                for n, d, dt in items:
                    print(f'    {n}: {d}')

        sorted_by_offset = sorted(all_tensors, key=lambda x: x[4])
        first_name, first_nd, first_dims, first_dt, first_off = sorted_by_offset[0]
        print(f'\nFirst data tensor: {first_name}, shape={first_dims}, dtype={first_dt}, offset={first_off}')
        
        f.seek(first_off)
        data = f.read(min(first_dims[0] * 4, 256))
        vals = struct.unpack('<' + 'i' * (len(data) // 4), data)
        print(f'First 32 int32 values: {vals[:32]}')
        if len(vals) == 64:
            print(f'Second 32: {vals[32:64]}')

        f.seek(0)
        magic2 = f.read(4)
        version2 = struct.unpack('<I', f.read(4))[0]
        n_tensors2 = struct.unpack('<Q', f.read(8))[0]
        n_kv2 = struct.unpack('<Q', f.read(8))[0]
        kv_data_end = f.tell()
        for i in range(n_kv2):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            val_type = struct.unpack('<i', f.read(4))[0]
            if val_type == 8:
                s_len = struct.unpack('<Q', f.read(8))[0]
                f.read(s_len)
            elif val_type == 5:
                f.read(4)
            elif val_type == 6:
                f.read(4)
            elif val_type == 0:
                f.read(1)
        
        print(f'KV data end pos: {kv_data_end}')
        tensor_meta_start = f.tell()
        print(f'Tensor metadata start: {tensor_meta_start}')
        print(f'Data start offset: {first_off}')
        print(f'Data start - metadata end = {first_off - tensor_meta_start} bytes of tensor metadata')
