"""Debug the build script."""

import torch
import struct
import numpy as np

data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
quantized = data['quantized']
e = quantized[0]

# Check what we're writing for layer 0
print("Layer 0:")
print(f"  U_packed: {e['U_packed'].shape}, nbytes={e['U_packed'].numel()}")
print(f"  Vt_packed: {e['Vt_packed'].shape}, nbytes={e['Vt_packed'].numel()}")
print(f"  S: {e['S'].shape}, nbytes={e['S'].numel()}")
print(f"  U_scale: {e['U_scale'].item():.6f}")
print(f"  Vt_scale: {e['Vt_scale'].item():.6f}")
print(f"  S_scale: {e['S_scale'].item():.6f}")
print(f"  U_shape: {e['U_shape']}")
print(f"  Vt_shape: {e['Vt_shape']}")

# Trace through the build
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

# Find first layer's tensors
layer0_tensors = [(n, a, d) for n, a, d in tensors if '.0.' in n]
print(f"\nLayer 0 tensors: {len(layer0_tensors)}")

# Compute offsets for layer 0
offset = 0
for name, arr, dtype in layer0_tensors:
    print(f"  {name}: offset={offset}, size={arr.nbytes}")
    offset += arr.nbytes

print(f"\nTotal layer 0 data: {offset} bytes")

# Now check what the GGUF should look like
# Build just the first part and verify

# Clear old file
import os
if os.path.exists('quantized/test_gguf.bin'):
    os.remove('quantized/test_gguf.bin')

# Simulate the build
with open('quantized/test_gguf.bin', 'wb') as f:
    # Header (24 bytes)
    f.write(struct.pack('<I', 0x46554747))
    f.write(struct.pack('<I', 3))
    f.write(struct.pack('<Q', len(tensors)))
    f.write(struct.pack('<Q', 8))

    # Metadata (302 bytes)
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

    for key, val in metadata_items:
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<I', len(key_bytes)))
        f.write(key_bytes)
        if isinstance(val, str):
            f.write(struct.pack('<I', 8))
            val_bytes = val.encode('utf-8')
            f.write(struct.pack('<Q', len(val_bytes)))
            f.write(val_bytes)
        elif isinstance(val, int):
            f.write(struct.pack('<I', 4))
            f.write(struct.pack('<i', val))

    header_and_meta = f.tell()
    print(f"\nAfter header+metadata: {header_and_meta}")

    # Write tensor info for layer 0 only
    current_offset = 0
    for name, arr, dtype in layer0_tensors:
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('<I', len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack('<Q', len(arr.shape)))
        for dim in arr.shape:
            f.write(struct.pack('<Q', dim))
        f.write(struct.pack('<I', dtype))
        f.write(struct.pack('<Q', current_offset))
        print(f"  Wrote tensor info for {name}, offset={current_offset}")
        current_offset += arr.nbytes

    tensor_info_end = f.tell()
    print(f"\nAfter tensor info: {tensor_info_end}, data starts at {f.tell()}")

    # Write tensor data for layer 0
    for name, arr, dtype in layer0_tensors:
        f.write(arr.tobytes())

    data_end = f.tell()
    print(f"After writing data: {data_end}")

# Verify
print("\nVerification:")
with open('quantized/test_gguf.bin', 'rb') as f:
    # Read header
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_meta = struct.unpack('<Q', f.read(8))[0]
    print(f"Magic: {hex(magic)}, Version: {version}, Tensors: {n_tensors}, Meta: {n_meta}")

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

    tensor_data_start = f.tell()
    print(f"Tensor data starts at: {tensor_data_start}")

    # Read tensor info for layer 0
    for name, arr, dtype in layer0_tensors:
        name_len = struct.unpack('<I', f.read(4))[0]
        tname = f.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<Q', f.read(8))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        tdtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]

        print(f"\n{tname}: offset={offset}, dtype={tdtype}")

        # Read data at that offset
        f.seek(tensor_data_start + offset)
        if 'scale' in tname.lower():
            data = f.read(4)
            val = struct.unpack('<f', data)[0]
            print(f"  Data: {data.hex()} = {val}")
        elif 'shape' in tname.lower():
            data = f.read(8)
            vals = struct.unpack('<ii', data)
            print(f"  Data: {data.hex()} = {vals}")

# Compare with expected
print("\nExpected:")
print(f"  U_scale: {struct.pack('<f', e['U_scale'].item()).hex()} = {e['U_scale'].item()}")
print(f"  Vt_scale: {struct.pack('<f', e['Vt_scale'].item()).hex()} = {e['Vt_scale'].item()}")
print(f"  U_shape: {np.array(e['U_shape'], dtype=np.int32).tobytes().hex()}")
print(f"  Vt_shape: {np.array(e['Vt_shape'], dtype=np.int32).tobytes().hex()}")