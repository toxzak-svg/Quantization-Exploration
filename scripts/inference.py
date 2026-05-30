import torch
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import struct
import numpy as np


def get_tensor_metadata(filepath: str) -> Dict[str, Dict]:
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = f.read(header_size)
        header_obj = json.loads(header)
    metadata = {}
    for name, info in header_obj.items():
        if name == '__metadata__':
            continue
        if not isinstance(info, dict) or 'shape' not in info:
            continue
        metadata[name] = {
            'shape': info['shape'],
            'dtype': info['dtype'],
            'offsets': info['data_offsets']
        }
    return metadata


def load_single_tensor(filepath: str, key: str) -> torch.Tensor:
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = f.read(header_size)
        header_obj = json.loads(header)
        info = header_obj[key]
        begin, end = info['data_offsets']
        f.seek(header_size + begin)
        data = f.read(end - begin)
    if info['dtype'] == 'BF16':
        arr = np.frombuffer(data, dtype=np.uint16).copy()
        return torch.from_numpy(arr).to(torch.uint16).view(torch.bfloat16).reshape(info['shape'])
    numpy_dtype = {'F16': np.float16, 'F32': np.float32,
                   'I32': np.int32, 'I16': np.int16,
                   'I8': np.int8, 'U8': np.uint8}.get(info['dtype'], np.float32)
    arr = np.frombuffer(data, dtype=numpy_dtype).copy()
    return torch.from_numpy(arr).reshape(info['shape'])


def unpack_5x8(packed: torch.Tensor, shape: List[int]) -> torch.Tensor:
    total_elements = shape[0] * shape[1]
    expected_packed = (total_elements + 4) // 5
    assert packed.numel() == expected_packed, \
        f"Expected {expected_packed} packed bytes, got {packed.numel()}"
    ternary_map = {0: -1, 1: 0, 2: 1}
    result = torch.zeros(total_elements, dtype=torch.float32)
    for i in range(total_elements):
        byte_idx = i // 5
        bit_offset = (i % 5) * 2
        val = (packed[byte_idx].item() >> bit_offset) & 0x03
        result[i] = ternary_map.get(val, 0.0)
    return result.reshape(shape)


def reconstruct_weight(q_entry: dict, device: str = 'cpu') -> torch.Tensor:
    U_packed = q_entry['U_packed']
    U_shape = list(q_entry['U_shape'])
    Vt_packed = q_entry['Vt_packed']
    Vt_shape = list(q_entry['Vt_shape'])
    S = q_entry['S'].float()
    scale_U = q_entry['scale_U'].float()
    scale_Vt = q_entry['scale_Vt'].float()
    scale_S = q_entry['scale_S'].float() if 'scale_S' in q_entry else torch.tensor(1.0)

    n_uv = 5
    U_flat = torch.zeros(U_shape[0] * U_shape[1], dtype=torch.float32)
    for i in range(U_shape[0] * U_shape[1]):
        v = (U_packed[i // n_uv].item() >> ((i % n_uv) * 2)) & 0x03
        U_flat[i] = {-1: -1.0, 0: 0.0, 1: 1.0, 2: 0.0}.get(v, 0.0)
    U = U_flat.reshape(U_shape) * scale_U

    Vt_flat = torch.zeros(Vt_shape[0] * Vt_shape[1], dtype=torch.float32)
    for i in range(Vt_shape[0] * Vt_shape[1]):
        v = (Vt_packed[i // n_uv].item() >> ((i % n_uv) * 2)) & 0x03
        Vt_flat[i] = {-1: -1.0, 0: 0.0, 1: 1.0, 2: 0.0}.get(v, 0.0)
    Vt = Vt_flat.reshape(Vt_shape) * scale_Vt

    S_q = S.clone()
    # S is int8 with values 0 or 1 (quantized 2-bit, only positive range used since sigmas >= 0)
    # Dequantize: S_float = S_quant * scale_S
    S_float = S_q * scale_S

    return torch.matmul(U * S_float.unsqueeze(0), Vt.to(torch.float32))


def build_weight_mapping(model_dir: Path) -> Tuple[List[Tuple[str, List[int]]], Dict[str, str]]:
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    weight_keys = []
    file_mapping = {}
    for sf in safetensor_files:
        metadata = get_tensor_metadata(str(sf))
        for k, info in metadata.items():
            if 'weight' not in k:
                continue
            if len(info['shape']) != 2:
                continue
            weight_keys.append((k, info['shape']))
            file_mapping[k] = str(sf)
    return weight_keys, file_mapping


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sub-1-Bit LLM Inference')
    parser.add_argument('--model-dir', type=str,
                        default=r'C:\Users\Zwmar\projects\sub1quant\models\llama-2-7b-hf')
    parser.add_argument('--quantized-pt', type=str,
                        default=r'C:\Users\Zwmar\projects\sub1quant\llama-2-7b-sub1bit.gguf.pt')
    parser.add_argument('--prompt', type=str, default='The future of AI is')
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    except Exception as e:
        print(f"AutoTokenizer failed ({e}), trying LlamaTokenizer with local tokenizer.model")
        tokenizer = LlamaTokenizer.from_pretrained(".", trust_remote_code=True)

    q_data = torch.load(args.quantized_pt, map_location='cpu', weights_only=True)
    quantized = q_data['quantized']
    config = q_data.get('config', {})
    print(f"Loaded {len(quantized)} quantized entries")

    weight_keys, file_mapping = build_weight_mapping(Path(args.model_dir))
    print(f"Found {len(weight_keys)} weight matrices in original model")

    name_to_idx = {}
    for idx, (name, shape) in enumerate(weight_keys):
        name_to_idx[name] = idx

    layer_idx_to_name = {v: k for k, v in name_to_idx.items()}

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    print("Replacing weights with low-rank factors...")
    replaced = 0
    for idx, q_entry in quantized.items():
        if idx not in layer_idx_to_name:
            print(f"  Warning: quantized entry {idx} not found in model weights")
            continue
        name = layer_idx_to_name[idx]
        try:
            W_recon = reconstruct_weight(q_entry, device)
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                module_name, attr = parts[0], parts[1]
                module = model.get_submodule(module_name)
                if hasattr(module, attr):
                    getattr(module, attr).data = W_recon.to(device=device, dtype=torch.float16 if device != 'cpu' else torch.float32)
                    replaced += 1
        except Exception as e:
            print(f"  Error replacing {name}: {e}")

    print(f"Replaced {replaced}/{len(quantized)} weights")

    print(f"\nGenerating with prompt: '{args.prompt}'")
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
    print(f"\nGenerated text:\n{generated}")


if __name__ == '__main__':
    main()
