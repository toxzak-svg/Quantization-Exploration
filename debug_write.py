"""Debug the write process."""

import torch
import struct
import numpy as np

data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
q = data['quantized']
e = q[0]

# What we expect to write
print("Layer 0 tensors:")
print(f"  U_packed: shape={e['U_packed'].shape}, nbytes={e['U_packed'].numel()}")
print(f"  Vt_packed: shape={e['Vt_packed'].shape}, nbytes={e['Vt_packed'].numel()}")
print(f"  S: shape={e['S'].shape}, nbytes={e['S'].numel()}")
print(f"  U_scale: {e['U_scale'].item()}")
print(f"  Vt_scale: {e['Vt_scale'].item()}")
print(f"  S_scale: {e['S_scale'].item()}")
print(f"  U_shape: {e['U_shape']}")
print(f"  Vt_shape: {e['Vt_shape']}")

# Build header calculation
tensors = []
for layer_idx, entry in q.items():
    tensors.append((f'model.layers.{layer_idx}.U_packed', entry['U_packed'].numpy(), 5))

# Just print first tensor info
name, arr, dtype = tensors[0]
print(f"\nFirst tensor: {name}")
print(f"  nbytes: {arr.nbytes}")
print(f"  shape: {arr.shape}")

# Calculate what current_offset should be
# After writing header (24) + metadata (302) + tensor info entries for all 316*8 tensors
# But we only have 1 tensor here...

# Let me compute expected offsets
U_packed_size = e['U_packed'].numel()
Vt_packed_size = e['Vt_packed'].numel()
S_size = e['S'].numel()

print(f"\nExpected layout:")
print(f"  U_packed at 0, size={U_packed_size}")
print(f"  Vt_packed at {U_packed_size}, size={Vt_packed_size}")
print(f"  S at {U_packed_size + Vt_packed_size}, size={S_size}")
print(f"  U_scale at {U_packed_size + Vt_packed_size + S_size}, size=4")
print(f"  Vt_scale at {U_packed_size + Vt_packed_size + S_size + 4}, size=4")
print(f"  S_scale at {U_packed_size + Vt_packed_size + S_size + 8}, size=4")
print(f"  U_shape at {U_packed_size + Vt_packed_size + S_size + 12}, size=8")
print(f"  Vt_shape at {U_packed_size + Vt_packed_size + S_size + 20}, size=8")

total_before_shapes = U_packed_size + Vt_packed_size + S_size + 12
print(f"\nU_shape should be at offset: {total_before_shapes}")
print(f"Vt_shape should be at offset: {total_before_shapes + 8}")

# Now write to a test file and verify
test_file = 'quantized/test_write.bin'

with open(test_file, 'wb') as f:
    # Simulate header (24 bytes) + metadata (302 bytes) = 326 bytes before tensor data
    f.write(b'\x00' * 326)

    # Now write tensors with offsets relative to this position
    current_offset = 0

    # U_packed
    print(f"\nWriting U_packed at current_offset={current_offset}")
    f.write(e['U_packed'].numpy().tobytes())
    current_offset += U_packed_size

    # Vt_packed
    print(f"Writing Vt_packed at current_offset={current_offset}")
    f.write(e['Vt_packed'].numpy().tobytes())
    current_offset += Vt_packed_size

    # S
    print(f"Writing S at current_offset={current_offset}")
    f.write(e['S'].numpy().tobytes())
    current_offset += S_size

    # U_scale
    print(f"Writing U_scale at current_offset={current_offset}")
    u_scale_bytes = struct.pack('<f', e['U_scale'].item())
    f.write(u_scale_bytes)
    current_offset += 4

    # Vt_scale
    print(f"Writing Vt_scale at current_offset={current_offset}")
    vt_scale_bytes = struct.pack('<f', e['Vt_scale'].item())
    f.write(vt_scale_bytes)
    current_offset += 4

    # S_scale
    print(f"Writing S_scale at current_offset={current_offset}")
    s_scale_bytes = struct.pack('<f', e['S_scale'].item())
    f.write(s_scale_bytes)
    current_offset += 4

    # U_shape
    print(f"Writing U_shape at current_offset={current_offset}")
    u_shape_bytes = np.array(e['U_shape'], dtype=np.int32).tobytes()
    f.write(u_shape_bytes)
    current_offset += 8

    # Vt_shape
    print(f"Writing Vt_shape at current_offset={current_offset}")
    vt_shape_bytes = np.array(e['Vt_shape'], dtype=np.int32).tobytes()
    f.write(vt_shape_bytes)

# Verify by reading
print("\nVerification:")
with open(test_file, 'rb') as f:
    # Skip header
    f.seek(326)

    # Read U_packed
    data = f.read(U_packed_size)
    print(f"U_packed: {len(data)} bytes")

    # Read Vt_packed
    data = f.read(Vt_packed_size)
    print(f"Vt_packed: {len(data)} bytes")

    # Read S
    data = f.read(S_size)
    print(f"S: {len(data)} bytes")

    # Read U_scale
    data = f.read(4)
    print(f"U_scale: {data.hex()} = {struct.unpack('<f', data)[0]}")

    # Read Vt_scale
    data = f.read(4)
    print(f"Vt_scale: {data.hex()} = {struct.unpack('<f', data)[0]}")

    # Read S_scale
    data = f.read(4)
    print(f"S_scale: {data.hex()} = {struct.unpack('<f', data)[0]}")

    # Read U_shape
    data = f.read(8)
    print(f"U_shape: {data.hex()} = {struct.unpack('<ii', data)}")

    # Read Vt_shape
    data = f.read(8)
    print(f"Vt_shape: {data.hex()} = {struct.unpack('<ii', data)}")

print("\nExpected:")
print(f"  U_scale: 0.3261 = {struct.pack('<f', 0.3261).hex()}")
print(f"  Vt_scale: 0.0791 = {struct.pack('<f', 0.0791).hex()}")
print(f"  S_scale: 192.0314 = {struct.pack('<f', 192.0314).hex()}")
print(f"  U_shape: (1536, 1155) = {np.array([1536, 1155], dtype=np.int32).tobytes().hex()}")
print(f"  Vt_shape: (1155, 6144) = {np.array([1155, 6144], dtype=np.int32).tobytes().hex()}")