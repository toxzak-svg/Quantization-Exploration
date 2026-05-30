import struct

with open('quantized/gemma-4-E2B-sub1bit-stream.gguf', 'rb') as f:
    data_start = 326

    # U_packed at offset 0
    f.seek(data_start + 0)
    data = f.read(16)
    print(f'U_packed[0:16]: {data.hex()}')

    # Vt_packed at offset 354816
    f.seek(data_start + 354816)
    data = f.read(16)
    print(f'Vt_packed[0:16]: {data.hex()}')

    # S at offset 1774080
    f.seek(data_start + 1774080)
    data = f.read(16)
    print(f'S[0:16]: {data.hex()}')

    # U_scale at offset 1775235
    f.seek(data_start + 1775235)
    data = f.read(4)
    val = struct.unpack('<f', data)[0]
    print(f'U_scale: {data.hex()} = {val}')

    # Vt_scale at offset 1775239
    f.seek(data_start + 1775239)
    data = f.read(4)
    val = struct.unpack('<f', data)[0]
    print(f'Vt_scale: {data.hex()} = {val}')

    # S_scale at offset 1775243
    f.seek(data_start + 1775243)
    data = f.read(4)
    val = struct.unpack('<f', data)[0]
    print(f'S_scale: {data.hex()} = {val}')

    # U_shape at offset 1775247
    f.seek(data_start + 1775247)
    data = f.read(8)
    vals = struct.unpack('<ii', data)
    print(f'U_shape: {data.hex()} = {vals}')

    # Vt_shape at offset 1775255
    f.seek(data_start + 1775255)
    data = f.read(8)
    vals = struct.unpack('<ii', data)
    print(f'Vt_shape: {data.hex()} = {vals}')

print("\nExpected:")
import numpy as np
import torch
data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
q = data['quantized']
e = q[0]
print(f"  U_scale: {struct.pack('<f', e['U_scale'].item()).hex()} = {e['U_scale'].item()}")
print(f"  Vt_scale: {struct.pack('<f', e['Vt_scale'].item()).hex()} = {e['Vt_scale'].item()}")
print(f"  S_scale: {struct.pack('<f', e['S_scale'].item()).hex()} = {e['S_scale'].item()}")
print(f"  U_shape: {np.array(e['U_shape'], dtype=np.int32).tobytes().hex()}")
print(f"  Vt_shape: {np.array(e['Vt_shape'], dtype=np.int32).tobytes().hex()}")