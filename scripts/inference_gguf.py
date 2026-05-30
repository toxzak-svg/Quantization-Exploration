"""Inference with the quantized sub-1-bit GGUF/PT checkpoint."""
import torch
import gc
import struct
import json
from pathlib import Path
from typing import Dict, List, Tuple


def unpack_5x8_vectorized(packed: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """Vectorized unpacking of 5 ternary values per uint8 byte."""
    total = shape[0] * shape[1]
    expected = (total + 4) // 5
    packed = packed[:expected]
    i = torch.arange(total, device=packed.device)
    byte_idx = i // 5
    bit_offset = (i % 5) * 2
    vals = (packed[byte_idx].to(torch.int32) >> bit_offset) & 0x03
    result = torch.where(vals == 0, -1.0, torch.where(vals == 1, 0.0, 1.0))
    return result.reshape(shape)


def reconstruct_weight(q_entry: dict, device: str = 'cpu') -> torch.Tensor:
    """Reconstruct a weight matrix from quantized low-rank factors."""
    U_packed = q_entry['U_packed'].to(device)
    U_shape = list(q_entry['U_shape'])
    S = q_entry['S'].to(device).float()
    Vt_packed = q_entry['Vt_packed'].to(device)
    Vt_shape = list(q_entry['Vt_shape'])
    scale_U = q_entry['scale_U'].to(device).float()
    scale_Vt = q_entry['scale_Vt'].to(device).float()
    scale_S = q_entry.get('scale_S', torch.tensor(1.0)).to(device).float()

    U = unpack_5x8_vectorized(U_packed, U_shape) * scale_U
    Vt = unpack_5x8_vectorized(Vt_packed, Vt_shape) * scale_Vt
    S_float = S * scale_S

    return torch.matmul(U * S_float.unsqueeze(0), Vt.to(torch.float32))


def load_quantized_pt(path: str) -> Tuple[Dict, Dict]:
    """Load quantized checkpoint from .gguf.pt file."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    return data['quantized'], data.get('config', {})


def build_weight_mapping(model_dir: Path) -> Tuple[List[Tuple[str, List[int]]], Dict[str, str]]:
    """Build (key, shape) list and file mapping from safetensor directory."""
    import numpy as np

    def get_metadata(fp):
        with open(fp, 'rb') as f:
            hs = struct.unpack('<Q', f.read(8))[0]
            return json.loads(f.read(hs))

    safetensor_files = sorted(Path(model_dir).glob("*.safetensors"))
    weight_keys = []
    file_mapping = {}
    for sf in safetensor_files:
        metadata = get_metadata(str(sf))
        for k, info in metadata.items():
            if k == '__metadata__':
                continue
            if 'weight' not in k:
                continue
            if len(info['shape']) != 2:
                continue
            weight_keys.append((k, info['shape']))
            file_mapping[k] = str(sf)
    return weight_keys, file_mapping


def replace_weights(model, quantized: Dict, weight_keys: List, file_mapping: Dict, device: str):
    """Replace matching weights with quantized low-rank reconstructions."""
    name_to_idx = {}
    for idx, (name, shape) in enumerate(weight_keys):
        name_to_idx[name] = idx

    replaced = 0
    for name, module in model.named_modules():
        if not hasattr(module, 'weight'):
            continue
        if name not in name_to_idx:
            continue
        idx = name_to_idx[name]
        if idx not in quantized:
            continue

        W_recon = reconstruct_weight(quantized[idx], device)
        with torch.no_grad():
            module.weight.data = W_recon.to(dtype=module.weight.dtype, device=module.weight.device)
        replaced += 1

    return replaced


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sub-1-Bit LLM Inference')
    parser.add_argument('--model-dir', default=r'C:\Users\Zwmar\projects\sub1quant\models\llama-2-7b-hf')
    parser.add_argument('--quantized-pt', default=r'C:\Users\Zwmar\projects\sub1quant\llama-2-7b-sub1bit.gguf.pt')
    parser.add_argument('--prompt', default='The future of AI is')
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    print(f"Loading quantized checkpoint: {args.quantized_pt}")
    quantized, config = load_quantized_pt(args.quantized_pt)
    print(f"Loaded {len(quantized)} quantized entries")

    print(f"Loading base model from {args.model_dir}...")
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    print("Building weight mapping...")
    weight_keys, file_mapping = build_weight_mapping(Path(args.model_dir))
    print(f"Found {len(weight_keys)} weight matrices")

    print("Replacing weights with quantized low-rank factors...")
    replaced = replace_weights(model, quantized, weight_keys, file_mapping, device)
    print(f"Replaced {replaced}/{len(quantized)} weights")

    print(f"\nGenerating...")
    print(f"Prompt: '{args.prompt}'")
    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{generated}")

    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
