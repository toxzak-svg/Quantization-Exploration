import sys, os, torch, gc
sys.path.insert(0, r'.\src')
os.chdir(r'C:\Users\Zwmar\projects\sub1quant')

# Check what we have and calculate compression stats
print("=== Current State Analysis ===\n")

# 1. Check model size
from quantize import SubOneBitQuantizer
quantizer = SubOneBitQuantizer(r'.\models\llama-2-7b-hf', device='cpu')
quantizer.load_model_weights()

total_params = 0
for idx in range(quantizer.total_weights):
    W = quantizer.load_weight(idx)
    total_params += W.shape[0] * W.shape[1]
    del W
    gc.collect()

print(f"Total model parameters: {total_params:,} ({total_params*2/1024/1024:.0f} MB @ FP16)")

# 2. Current quantized stats
q_data = torch.load(r'.\llama-2-7b-sub1bit.gguf.pt', map_location='cpu', weights_only=True)
quantized = q_data['quantized']
print(f"\nExisting quantized: {len(quantized)} entries")
print(f"Ranks: {min(q['rank'] for q in quantized.values())} - {max(q['rank'] for q in quantized.values())}")
avg_rank = sum(q['rank'] for q in quantized.values()) / len(quantized)
print(f"Average rank: {avg_rank:.0f}")

# 3. Calculate total bits for original quantization
total_bits = 0
for entry in quantized.values():
    U_shape = tuple(entry['U_shape'])
    Vt_shape = tuple(entry['Vt_shape'])
    S_shape = entry['S'].shape
    n_U = U_shape[0] * U_shape[1]
    n_S = S_shape[0]
    n_Vt = Vt_shape[0] * Vt_shape[1]
    packed_U = entry['U_packed'].numel() * 5
    packed_Vt = entry['Vt_packed'].numel() * 5
    bits_S = n_S * 2
    total_bits += packed_U + bits_S + packed_Vt

print(f"Current bits/param: {total_bits/(total_params*2):.3f}")  # *2 because packed 5bits in 8bits
print(f"Current size: {total_bits/8/1024/1024:.1f} MB")

# 4. What if we used rank 16?
print("\n=== Projected Compression with Rank 16 ===")
for rank in [8, 16, 32, 64]:
    bits_per_param = (0.5 + 2.0/rank + 0.5)  # U bits + S bits + Vt bits, normalized
    size_bits = total_params * bits_per_param
    print(f"Rank {rank:2d}: {bits_per_param:.3f} bits/param -> {size_bits/8/1024/1024:.1f} MB")
    if rank == 16:
        print(f"  Energy captured @ rank 16: see analysis above (~3-90% depending on layer)")

# 5. Check transforms
print("\n=== Learned Transforms ===")
transforms = torch.load(r'.\quantized\transforms.pt', map_location='cpu', weights_only=False)
print(f"Codebook dim: {transforms['codebook_dim']}")
print(f"Layers: {len(transforms['layers'])}")
print(f"Layer sizes: {len(transforms['layer_sizes'])}")
for i, ls in enumerate(transforms['layer_sizes'][:5]):
    print(f"  {i}: {ls}")

print("\n=== Next Steps ===")
print("1. Need to re-quantize with proper low rank (rank 16-32)")
print("2. Apply learned transforms before SVD for better energy concentration")
print("3. Then evaluate perplexity")