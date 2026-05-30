import sys, os, torch, gc
sys.path.insert(0, r'.\src')
os.chdir(r'C:\Users\Zwmar\projects\sub1quant')
from quantize import SubOneBitQuantizer

quantizer = SubOneBitQuantizer(r'.\models\llama-2-7b-hf', device='cpu')
quantizer.load_model_weights()

# Process just 4 layers with rank 16
output_dir = r'.\quantized\rank16_test'
os.makedirs(output_dir, exist_ok=True)

for idx in range(4):
    W = quantizer.load_weight(idx)
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    r = 16

    factor = {
        'U': U[:, :r].half().numpy(),
        'S': S[:r].half().numpy(),
        'Vt': Vt[:r, :].half().numpy(),
        'rank': r,
        'original_shape': list(W.shape),
        'energy_captured': float((S[:r]**2).sum()/(S**2).sum())
    }

    torch.save(factor, f'{output_dir}\\layer_{idx:04d}.pt')
    print(f'Layer {idx}: rank={r}, energy={factor["energy_captured"]:.4f}, shape={W.shape}')

    del U, S, Vt, W, factor
    gc.collect()
print('Done')