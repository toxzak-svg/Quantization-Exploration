import torch
import gc
import argparse
import json
import struct
from pathlib import Path
import sys
from typing import Callable, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.groupwise_int4 import dequantize_groupwise_int4
from src.quantization import ternary_unpack, sigma_dequantize


def _as_tensor(value, device: str, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _shape_tuple(shape) -> Optional[tuple]:
    if shape is None:
        return None
    if isinstance(shape, torch.Tensor):
        shape = shape.tolist()
    return tuple(int(dim) for dim in shape)


def quantized_entry_shape(q_entry: dict) -> Optional[tuple]:
    shape = q_entry.get('original_shape', q_entry.get('orig_shape'))
    if shape is not None:
        return _shape_tuple(shape)
    if 'U_shape' in q_entry and 'Vt_shape' in q_entry:
        u_shape = _shape_tuple(q_entry['U_shape'])
        vt_shape = _shape_tuple(q_entry['Vt_shape'])
        return (u_shape[0], vt_shape[1])
    return None


def normalize_checkpoint_weight_keys(weight_keys: Optional[Iterable]) -> list[tuple[str, tuple]]:
    if not weight_keys:
        return []

    normalized = []
    for item in weight_keys:
        if isinstance(item, dict):
            key = item.get('key') or item.get('name')
            shape = item.get('shape') or item.get('original_shape') or item.get('orig_shape')
        else:
            key, shape = item
        if key:
            normalized.append((key, _shape_tuple(shape)))
    return normalized


def load_model_weight_keys(model_dir: str | Path) -> list[tuple[str, tuple]]:
    """Recover quantization key order from local safetensors metadata."""
    model_dir = Path(model_dir)
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    weight_keys = []

    for safetensor_path in safetensor_files:
        with open(safetensor_path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_size))

        for key, info in header.items():
            if key == '__metadata__' or not isinstance(info, dict):
                continue
            shape = info.get('shape')
            if 'weight' not in key or shape is None or len(shape) != 2:
                continue
            if any(x in key for x in [
                'lm_head',
                'embed_tokens',
                'norm',
                'audio_tower',
                'vision_tower',
                'embed_vision',
            ]):
                continue
            weight_keys.append((key, _shape_tuple(shape)))

    language_model_keys = [item for item in weight_keys if 'language_model' in item[0]]
    return language_model_keys or weight_keys


def resolve_quantized_weight_key(
    entry_index,
    q_entry: dict,
    fallback_weight_keys: Optional[list[tuple[str, tuple]]] = None,
) -> str:
    key = q_entry.get('key')
    if key:
        return key

    if fallback_weight_keys:
        try:
            fallback_index = int(entry_index)
        except (TypeError, ValueError) as exc:
            raise KeyError(
                f"Quantized entry {entry_index!r} has no key and cannot be mapped by index"
            ) from exc

        if fallback_index >= len(fallback_weight_keys):
            raise KeyError(
                f"Quantized entry {entry_index!r} has no key and exceeds "
                f"{len(fallback_weight_keys)} fallback model weights"
            )

        key, expected_shape = fallback_weight_keys[fallback_index]
        entry_shape = quantized_entry_shape(q_entry)
        if entry_shape is not None and expected_shape is not None and entry_shape != expected_shape:
            raise ValueError(
                f"Legacy quantized entry {entry_index!r} maps to {key}, but "
                f"checkpoint shape {entry_shape} != model shape {expected_shape}"
            )
        return key

    raise KeyError(
        f"Quantized entry {entry_index!r} has no source key. Pass a local "
        "model directory with safetensors metadata or use a checkpoint that stores keys."
    )


def reconstruct_weight(q_entry: dict, device: str = 'cpu') -> torch.Tensor:
    U_scale = _as_tensor(q_entry['U_scale'], device)
    Vt_scale = _as_tensor(q_entry['Vt_scale'], device)
    S_scale = _as_tensor(q_entry['S_scale'], device)
    U = ternary_unpack(q_entry['U_packed'].to(device), q_entry['U_shape']).float() * U_scale
    Vt = ternary_unpack(q_entry['Vt_packed'].to(device), q_entry['Vt_shape']).float() * Vt_scale
    S = sigma_dequantize(q_entry['S'].to(device), S_scale)
    return torch.matmul(U * S.unsqueeze(0), Vt)


def reconstruct_quantized_entry(q_entry: dict, device: str = 'cpu') -> torch.Tensor:
    if q_entry.get('format') == 'groupwise_int4' or 'packed_int4' in q_entry:
        return dequantize_groupwise_int4(q_entry, device)

    if {'U_packed', 'Vt_packed', 'S'}.issubset(q_entry):
        return reconstruct_weight(q_entry, device)

    if 'packed' in q_entry:
        shape = q_entry['orig_shape']
        scale = _as_tensor(q_entry['scale'], device)
        return ternary_unpack(q_entry['packed'].to(device), shape).float() * scale

    if 'q' in q_entry:
        num_bits = int(q_entry.get('num_bits', 0))
        qmax = 2 ** (num_bits - 1) - 1
        if qmax <= 0:
            raise ValueError(f"Unsupported num_bits for magnitude checkpoint: {num_bits}")

        q = q_entry['q'].to(device).float()
        shape = quantized_entry_shape(q_entry)
        if shape is not None and q.numel() == shape[0] * shape[1]:
            q = q.reshape(shape)

        scale = q_entry['scale']
        if q_entry.get('per_channel'):
            scale = _as_tensor(scale, device).reshape(-1, 1)
        else:
            scale = _as_tensor(scale, device)
        return q * scale / qmax

    raise ValueError(f"Unknown quantized entry format: {sorted(q_entry.keys())}")


def build_model_weight_map(model) -> dict[str, object]:
    weight_map = {}
    for name, module in model.named_modules():
        if getattr(module, 'weight', None) is None:
            continue
        weight_key = f"{name}.weight" if name else "weight"
        weight_map[weight_key] = module
    return weight_map


def is_expected_missing_shared_kv_weight(key: str, weight_map: dict[str, object]) -> bool:
    if key.endswith('.self_attn.k_proj.weight'):
        q_key = key.replace('.k_proj.weight', '.q_proj.weight')
        o_key = key.replace('.k_proj.weight', '.o_proj.weight')
        return q_key in weight_map or o_key in weight_map
    if key.endswith('.self_attn.v_proj.weight'):
        q_key = key.replace('.v_proj.weight', '.q_proj.weight')
        o_key = key.replace('.v_proj.weight', '.o_proj.weight')
        return q_key in weight_map or o_key in weight_map
    return False


def apply_quantized_weights(
    model,
    quantized: dict,
    device: str = 'cpu',
    model_dir: str | Path | None = None,
    checkpoint_weight_keys: Optional[Iterable] = None,
    reconstruct_fn: Callable[[dict, str], torch.Tensor] = reconstruct_quantized_entry,
    strict: bool = True,
) -> dict:
    fallback_weight_keys = normalize_checkpoint_weight_keys(checkpoint_weight_keys)
    if not fallback_weight_keys and model_dir is not None:
        fallback_weight_keys = load_model_weight_keys(model_dir)

    weight_map = build_model_weight_map(model)
    stats = {'replaced': 0, 'skipped': [], 'missing': [], 'shape_mismatches': []}

    for entry_index, q_entry in quantized.items():
        key = resolve_quantized_weight_key(entry_index, q_entry, fallback_weight_keys)
        module = weight_map.get(key)
        if module is None:
            if is_expected_missing_shared_kv_weight(key, weight_map):
                stats['skipped'].append(key)
                continue
            message = f"No model module found for quantized weight {key}"
            if strict:
                raise KeyError(message)
            stats['missing'].append(message)
            continue

        reconstructed = reconstruct_fn(q_entry, device)
        target_shape = tuple(module.weight.shape)
        if tuple(reconstructed.shape) != target_shape:
            message = (
                f"Shape mismatch for {key}: reconstructed {tuple(reconstructed.shape)} "
                f"!= model {target_shape}"
            )
            if strict:
                raise ValueError(message)
            stats['shape_mismatches'].append(message)
            continue

        with torch.no_grad():
            module.weight.data = reconstructed.to(
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
        stats['replaced'] += 1

    return stats


def eval_perplexity(model, tokenizer, wikitext_path: str, device: str,
                    max_length: int = 512, stride: int = 512):
    with open(wikitext_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Tokenizing...")
    encodings = tokenizer(text, return_tensors='pt')
    encodings = {k: v.to(device) for k, v in encodings.items()}
    seq_len = encodings['input_ids'].shape[1]
    print(f"  Sequence length: {seq_len} tokens")

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings['input_ids'][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc >= seq_len:
            break

    avg_nll = torch.stack(nlls).sum() / seq_len
    perplexity = torch.exp(avg_nll).item()
    return perplexity, {'n_chunks': len(nlls), 'seq_len': seq_len}


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized model perplexity")
    parser.add_argument('--quantized-pt', default='quantized/gemma-4-E2B-sub1bit.pt',
                        help='Path to quantized .pt checkpoint')
    parser.add_argument('--model-dir', default='models/gemma-4-E2B',
                        help='Path to base model directory')
    parser.add_argument('--wikitext', default='data/wiki.test.txt',
                        help='Path to WikiText test file')
    parser.add_argument('--device', default=None,
                        help='Device (auto-detect if not set)')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=512)
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wikitext_path = Path(args.wikitext)
    if not wikitext_path.exists():
        print(f"WikiText not found: {wikitext_path}")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("QUANTIZED MODEL PERPLEXITY EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Quantized: {args.quantized_pt}")
    print(f"Base model: {args.model_dir}")
    print(f"WikiText: {args.wikitext}")
    print()

    # 1. Load quantized checkpoint
    print("[1] Loading quantized checkpoint...")
    q_data = torch.load(args.quantized_pt, map_location='cpu', weights_only=True)
    quantized = q_data['quantized']
    print(f"  {len(quantized)} quantized entries")
    print()

    # 2. Load base model
    print("[2] Loading base model...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map=device,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    model.eval()
    print()

    # 3. Reconstruct weights
    print("[3] Applying quantized weights...")
    apply_stats = apply_quantized_weights(
        model,
        quantized,
        device=device,
        model_dir=args.model_dir,
        checkpoint_weight_keys=q_data.get('weight_keys'),
    )
    print(f"  Replaced {apply_stats['replaced']}/{len(quantized)} weights")
    if apply_stats['skipped']:
        print(f"  Skipped {len(apply_stats['skipped'])} shared-KV checkpoint entries")
    print()

    # 4. Evaluate perplexity
    print("[4] Evaluating perplexity...")
    ppl, stats = eval_perplexity(
        model, tokenizer, str(wikitext_path), device,
        max_length=args.max_length, stride=args.stride
    )
    print()

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Perplexity: {ppl:.4f}")
    print(f"  Chunks: {stats['n_chunks']}")
    print(f"  Target: <= 10.5")
    status = "PASS" if ppl <= 10.5 else "FAIL"
    print(f"  Status: {status}")
    print("=" * 60)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
