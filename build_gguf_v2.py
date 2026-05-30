"""Build GGUF with correct sequential layout."""

import torch
import struct
import numpy as np

def build_sub1bit_gguf():
    data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
    quantized = data['quantized']

    # Build tensor list
    tensors = []
    for layer_idx, entry in quantized.items():
        tensors.append((f'model.layers.{layer_idx}.U_packed', entry['U_packed'].numpy(), 5))
        tensors.append((f'model.layers.{layer_idx}.Vt_packed', entry['Vt_packed'].numpy(), 5))
        tensors.append((f'model.layers.{layer_idx}.S', entry['S'].numpy(), 6))
        tensors.append((f'model.layers.{layer_idx}.U_scale', np.array([entry['U_scale'].item()], dtype=np.float32), 2))
        tensors.append((f'model.layers.{layer_idx}.Vt_scale', np.array([entry['Vt_scale'].item()], dtype=np.float32), 2))
        tensors.append((f'model.layers.{layer_idx}.S_scale', np.array([entry['S_scale'].item()], dtype=np.float32), 2))
        tensors.append((f'model.layers.{layer_idx}.U_shape', np.array(entry['U_shape'], dtype=np.int32), 4))
        tensors.append((f'model.layers.{layer_idx}.Vt_shape', np.array(entry['Vt_shape'], dtype=np.int32), 4))

    output_path = 'quantized/gemma-4-E2B-sub1bit-stream.gguf'

    # Metadata
    metadata_items = [
        ('general.architecture', 'gemma'),
        ('general.name', 'gemma-4-E2B-sub1bit'),
        ('quantization.version', 1),
        ('quantization.type', 'sub1bit_lowrank'),
        ('gemma.embedding_dimension', 1536),
        ('gemma.hidden_dimension', 6144),
        ('gemma.num_layers', len(quantized)),
        ('gemma.num_local_experts', 0),
    ]

    # Calculate exact header size
    header = struct.pack('<I', 0x46554747) + struct.pack('<I', 3) + struct.pack('<Q', len(tensors)) + struct.pack('<Q', len(metadata_items))
    header_size = len(header)

    metadata_bytes = b''
    for key, val in metadata_items:
        key_bytes = key.encode('utf-8')
        metadata_bytes += struct.pack('<I', len(key_bytes)) + key_bytes
        if isinstance(val, str):
            metadata_bytes += struct.pack('<I', 8)
            val_bytes = val.encode('utf-8')
            metadata_bytes += struct.pack('<Q', len(val_bytes)) + val_bytes
        elif isinstance(val, int):
            metadata_bytes += struct.pack('<I', 4)
            metadata_bytes += struct.pack('<i', val)

    tensor_info_bytes = b''
    for name, arr, dtype in tensors:
        name_bytes = name.encode('utf-8')
        tensor_info_bytes += struct.pack('<I', len(name_bytes)) + name_bytes
        tensor_info_bytes += struct.pack('<Q', len(arr.shape))
        for dim in arr.shape:
            tensor_info_bytes += struct.pack('<Q', dim)
        tensor_info_bytes += struct.pack('<I', dtype)
        tensor_info_bytes += struct.pack('<Q', 0)  # placeholder

    data_start = header_size + len(metadata_bytes) + len(tensor_info_bytes)

    print(f"Header: {header_size}, Metadata: {len(metadata_bytes)}, TensorInfo: {len(tensor_info_bytes)}, DataStart: {data_start}")

    # Write file
    with open(output_path, 'wb') as f:
        # Header
        f.write(header)

        # Metadata
        f.write(metadata_bytes)

        # Tensor info with correct sequential offsets
        current_offset = 0
        tensor_info_with_offsets = []
        for name, arr, dtype in tensors:
            name_bytes = name.encode('utf-8')
            tensor_info = struct.pack('<I', len(name_bytes)) + name_bytes
            tensor_info += struct.pack('<Q', len(arr.shape))
            for dim in arr.shape:
                tensor_info += struct.pack('<Q', dim)
            tensor_info += struct.pack('<I', dtype)
            tensor_info += struct.pack('<Q', current_offset)
            tensor_info_with_offsets.append(tensor_info)
            current_offset += arr.nbytes

        # Write all tensor info
        for ti in tensor_info_with_offsets:
            f.write(ti)

        # Write all tensor data
        for name, arr, dtype in tensors:
            f.write(arr.tobytes())

    print(f"Written: {os.path.getsize(output_path) / 1e6:.1f} MB")

    # Verify
    print("\nVerification:")
    with open(output_path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_meta = struct.unpack('<Q', f.read(8))[0]

        # Skip metadata
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

        data_start_actual = f.tell()
        print(f"Actual data start: {data_start_actual}")

        # Find and verify layer 0 tensors
        layer0_tensors = {}
        for i in range(n_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<Q', f.read(8))[0]
            shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]

            if '.0.' in name:
                layer0_tensors[name] = (shape, dtype, offset)

        # Read and verify data
        for name, (shape, dtype, offset) in layer0_tensors.items():
            f.seek(data_start_actual + offset)
            if 'scale' in name.lower():
                data = f.read(4)
                val = struct.unpack('<f', data)[0]
                print(f"  {name}: offset={offset}, value={val:.6f}")
            elif 'shape' in name.lower():
                data = f.read(8)
                vals = struct.unpack('<ii', data)
                print(f"  {name}: offset={offset}, value={vals}")

import os
build_sub1bit_gguf()