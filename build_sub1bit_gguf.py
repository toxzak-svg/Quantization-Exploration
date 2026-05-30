"""
Build GGUF with correct offset handling.
Offsets in GGUF tensor info are relative to the start of tensor data.
"""

import torch
import struct
import numpy as np
import os


def build_sub1bit_gguf(quantized_pt_path: str, output_path: str):
    """Build GGUF from sub1bit quantized model."""

    print(f"Loading {quantized_pt_path}...")
    data = torch.load(quantized_pt_path, map_location='cpu', weights_only=True)
    quantized = data.get('quantized', {})

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

    # Calculate header size to determine where tensor data starts
    # Header: magic(4) + version(4) + n_tensors(8) + n_metadata(8) = 24
    # Metadata: key-value pairs
    # Tensor info: name_len(4) + name + n_dims(8) + dims + dtype(4) + offset(8)

    # Calculate sizes
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

    metadata_size = 0
    for key, val in metadata_items:
        metadata_size += 4 + len(key) + 4  # key_len + key + type
        if isinstance(val, str):
            metadata_size += 8 + len(val)  # string has length prefix
        elif isinstance(val, int):
            metadata_size += 4

    tensor_info_size = 0
    for name, arr, dtype in tensors:
        tensor_info_size += 4 + len(name.encode('utf-8')) + 8 + len(arr.shape) * 8 + 4 + 8

    tensor_data_start = 24 + metadata_size + tensor_info_size
    print(f"Header+metadata: {24 + metadata_size}, Tensor info: {tensor_info_size}, Data starts at: {tensor_data_start}")

    # Write file
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', 0x46554747))  # "GGUF"
        f.write(struct.pack('<I', 3))  # version
        f.write(struct.pack('<Q', len(tensors)))
        f.write(struct.pack('<Q', len(metadata_items)))

        # Metadata
        for key, val in metadata_items:
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(key_bytes)))
            f.write(key_bytes)
            if isinstance(val, str):
                f.write(struct.pack('<I', 8))  # string type
                val_bytes = val.encode('utf-8')
                f.write(struct.pack('<Q', len(val_bytes)))
                f.write(val_bytes)
            elif isinstance(val, int):
                f.write(struct.pack('<I', 4))  # int32 type
                f.write(struct.pack('<i', val))

        # Tensor info with RELATIVE offsets (from tensor_data_start)
        current_offset = 0
        for name, arr, dtype in tensors:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<Q', len(arr.shape)))
            for dim in arr.shape:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', dtype))
            f.write(struct.pack('<Q', current_offset))  # RELATIVE offset
            current_offset += arr.nbytes

        # Tensor data
        for name, arr, dtype in tensors:
            f.write(arr.tobytes())

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Written GGUF: {size_mb:.1f} MB")

    # Verify by reading back
    print("\nVerification:")
    with open(output_path, 'rb') as f:
        # Header
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

        actual_tensor_data_start = f.tell()
        print(f"Actual tensor data starts at: {actual_tensor_data_start}")

        # Read tensor info
        tensor_infos = []
        for i in range(n_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<Q', f.read(8))[0]
            shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append((name, shape, dtype, offset))

        # Check U_shape for layer 0
        for name, shape, dtype, offset in tensor_infos:
            if name == 'model.layers.0.U_shape':
                print(f"\n{name}: offset={offset}")
                absolute_pos = actual_tensor_data_start + offset
                f.seek(absolute_pos)
                data = f.read(8)
                print(f"  At absolute {absolute_pos}: {data.hex()}")
                values = struct.unpack('<ii', data)
                print(f"  As int32[2]: {values}")
                expected = (1536, 1155)
                print(f"  Expected: {expected}, Match: {values == expected}")

            if name == 'model.layers.0.Vt_shape':
                print(f"\n{name}: offset={offset}")
                absolute_pos = actual_tensor_data_start + offset
                f.seek(absolute_pos)
                data = f.read(8)
                print(f"  At absolute {absolute_pos}: {data.hex()}")
                values = struct.unpack('<ii', data)
                print(f"  As int32[2]: {values}")
                expected = (1155, 6144)
                print(f"  Expected: {expected}, Match: {values == expected}")

            if name == 'model.layers.0.U_scale':
                print(f"\n{name}: offset={offset}")
                absolute_pos = actual_tensor_data_start + offset
                f.seek(absolute_pos)
                data = f.read(4)
                print(f"  At absolute {absolute_pos}: {data.hex()}")
                value = struct.unpack('<f', data)[0]
                print(f"  As float32: {value}")
                print(f"  Expected: 0.3261")

    return size_mb


if __name__ == '__main__':
    build_sub1bit_gguf('quantized/gemma-4-E2B-sub1bit.pt', 'quantized/gemma-4-E2B-sub1bit-stream.gguf')