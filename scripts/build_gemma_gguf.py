import torch
import struct
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_LOWRANK_UV_1BIT = 42
GGML_TYPE_LOWRANK_S_2BIT = 43


def write_string(f, s):
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_kv(f, key, value):
    write_string(f, key)
    if isinstance(value, str):
        f.write(struct.pack('<i', 8))
        write_string(f, value)
    elif isinstance(value, bool):
        f.write(struct.pack('<i', 7))
        f.write(struct.pack('<B', 1 if value else 0))
    elif isinstance(value, int):
        f.write(struct.pack('<i', 5))
        f.write(struct.pack('<i', value))
    elif isinstance(value, float):
        f.write(struct.pack('<i', 6))
        f.write(struct.pack('<f', value))
    else:
        raise ValueError(f"Unsupported KV type: {type(value)}")


def get_safetensor_metadata(filepath):
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(header_size))


def discover_gemma_weights(model_dir):
    safetensor_files = sorted(Path(model_dir).glob("*.safetensors"))
    weight_keys = []
    file_mapping = {}
    for sf in safetensor_files:
        metadata = get_safetensor_metadata(str(sf))
        for k, info in metadata.items():
            if k == '__metadata__':
                continue
            if 'weight' not in k or len(info['shape']) != 2:
                continue
            if 'lm_head' in k or 'embed_tokens' in k or 'norm' in k:
                continue
            if 'audio_tower' in k or 'vision_tower' in k or 'embed_vision' in k:
                continue
            if 'language_model' not in k:
                continue
            weight_keys.append((k, info['shape']))
            file_mapping[k] = str(sf)
    return weight_keys, file_mapping


def copy_tensor_to_gguf(sf_path, key, gguf_file, offset):
    with open(sf_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        info = header[key]
        begin, end = info['data_offsets']
        f.seek(8 + header_size + begin)
        data = f.read(end - begin)
    gguf_file.seek(offset)
    gguf_file.write(data)
    return len(data)


def build_gemma_gguf(model_dir, quantized_pt, output_path):
    model_dir = Path(model_dir)
    print(f"Loading model from {model_dir}")
    print(f"Loading quantized data from {quantized_pt}")

    q_data = torch.load(quantized_pt, map_location='cpu', weights_only=True)
    quantized = q_data['quantized']

    with open(model_dir / 'config.json') as f:
        model_config = json.load(f)

    text_config = model_config.get('text_config', {})
    hidden_size = text_config.get('hidden_size', 1536)
    n_layers = text_config.get('num_hidden_layers', 35)
    n_heads = text_config.get('num_attention_heads', 8)
    n_kv_heads = text_config.get('num_key_value_heads', 1)
    inter_size = text_config.get('intermediate_size', 6144)
    vocab_size = text_config.get('vocab_size', 262144)
    sliding_window = text_config.get('sliding_window', 512)
    max_position_embeddings = text_config.get('max_position_embeddings', 131072)

    weight_keys, file_mapping = discover_gemma_weights(model_dir)
    print(f"Discovered {len(weight_keys)} weight matrices")

    all_metadata = {}
    for sf in sorted(Path(model_dir).glob("*.safetensors")):
        meta = get_safetensor_metadata(str(sf))
        for k, v in meta.items():
            if k != '__metadata__':
                all_metadata[k] = (str(sf), v)

    norm_keys = [k for k in all_metadata if 'norm' in k and 'weight' in k and 'language_model' in k]
    embed_name = 'model.language_model.embed_tokens.weight'
    lm_name = 'model.language_model.lm_head.weight'

    tensor_list = []

    for idx, (wk_name, wk_shape) in enumerate(weight_keys):
        if idx in quantized:
            qe = quantized[idx]
            u_nbytes = qe['U_packed'].numpy().nbytes
            s_nbytes = qe['S'].numpy().astype(np.int8).nbytes
            vt_nbytes = qe['Vt_packed'].numpy().nbytes
            tensor_list.append((f'lowrank.{idx}.U', 2, list(qe['U_shape']), GGML_TYPE_LOWRANK_UV_1BIT, u_nbytes, 'U_packed', idx))
            tensor_list.append((f'lowrank.{idx}.S', 1, [qe['S'].shape[0]], GGML_TYPE_LOWRANK_S_2BIT, s_nbytes, 'S', idx))
            tensor_list.append((f'lowrank.{idx}.Vt', 2, list(qe['Vt_shape']), GGML_TYPE_LOWRANK_UV_1BIT, vt_nbytes, 'Vt_packed', idx))

    for idx, (wk_name, wk_shape) in enumerate(weight_keys):
        if idx not in quantized:
            sf_path = file_mapping[wk_name]
            info = all_metadata[wk_name][1]
            dtype_str = info['dtype']
            ggml_type = GGML_TYPE_F16 if dtype_str in ['F16', 'BF16'] else GGML_TYPE_F32
            shape = info['shape']
            total_elems = np.prod(shape)
            elem_size = 2 if ggml_type == GGML_TYPE_F16 else 4
            tensor_list.append((wk_name, len(shape), shape, ggml_type, total_elems * elem_size, 'safetensor', sf_path, wk_name))

    for name in norm_keys:
        sf_path, info = all_metadata[name]
        shape = info['shape']
        total_elems = np.prod(shape)
        ggml_type = GGML_TYPE_F16 if info['dtype'] in ['F16', 'BF16'] else GGML_TYPE_F32
        elem_size = 2 if ggml_type == GGML_TYPE_F16 else 4
        tensor_list.append((name, len(shape), shape, ggml_type, total_elems * elem_size, 'safetensor', sf_path, name))

    if embed_name in all_metadata:
        sf_path, info = all_metadata[embed_name]
        shape = info['shape']
        total_elems = np.prod(shape)
        ggml_type = GGML_TYPE_F16 if info['dtype'] in ['F16', 'BF16'] else GGML_TYPE_F32
        elem_size = 2 if ggml_type == GGML_TYPE_F16 else 4
        tensor_list.append((embed_name, len(shape), shape, ggml_type, total_elems * elem_size, 'safetensor', sf_path, embed_name))

    if lm_name in all_metadata:
        sf_path, info = all_metadata[lm_name]
        shape = info['shape']
        total_elems = np.prod(shape)
        ggml_type = GGML_TYPE_F16 if info['dtype'] in ['F16', 'BF16'] else GGML_TYPE_F32
        elem_size = 2 if ggml_type == GGML_TYPE_F16 else 4
        tensor_list.append((lm_name, len(shape), shape, ggml_type, total_elems * elem_size, 'safetensor', sf_path, lm_name))

    print(f"Total tensors: {len(tensor_list)}")

    with open(output_path, 'wb') as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))

        real_tensors = [t for t in tensor_list if len(t) >= 5]
        n_tensors = len(real_tensors)
        n_kv_pairs = 14 + 1

        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', n_kv_pairs))

        kv_pairs = [
            ('general.architecture', 'gemma'),
            ('general.name', 'Gemma-4-E2B-Sub1Bit'),
            ('gemma.context_length', max_position_embeddings),
            ('gemma.embedding_length', hidden_size),
            ('gemma.block_count', n_layers),
            ('gemma.feed_forward_length', inter_size),
            ('gemma.attention.head_count', n_heads),
            ('gemma.attention.head_count_kv', n_kv_heads),
            ('gemma.attention.sliding_window', sliding_window),
            ('gemma.rope.freq_base', 1000000.0),
            ('gemma.vocab_size', vocab_size),
            ('general.file_type', 0),
            ('quantization.version', 1),
        ]
        for key, value in kv_pairs:
            write_kv(f, key, value)

        write_string(f, 'lowrank.n_quantized')
        f.write(struct.pack('<i', 5))
        f.write(struct.pack('<i', len(quantized)))

        data_start = f.tell()
        for t in real_tensors:
            data_start += 8 + len(t[0].encode('utf-8')) + 4 + 8 * t[1] + 4 + 8
        data_start = (data_start + 31) & ~31

        offset = data_start
        offsets = []
        for t in real_tensors:
            offsets.append(offset)
            nbytes = t[4]
            offset += nbytes

        for i, t in enumerate(real_tensors):
            name = t[0]
            ndim = t[1]
            shape = t[2]
            gtype = t[3]
            name_encoded = name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_encoded)))
            f.write(name_encoded)
            f.write(struct.pack('<I', ndim))
            for d in shape:
                f.write(struct.pack('<q', d))
            f.write(struct.pack('<I', gtype))
            f.write(struct.pack('<Q', offsets[i]))

        padding = b'\0' * (data_start - f.tell())
        f.write(padding)

        for i, t in enumerate(real_tensors):
            if t[5] == 'safetensor':
                copy_tensor_to_gguf(t[6], t[7], f, offsets[i])
            else:
                q_idx = t[6]
                qe = quantized[q_idx]
                data = qe[t[5]].numpy()
                if t[5] == 'S':
                    data = data.astype(np.int8)
                f.write(data.tobytes())

        aligned_end = (f.tell() + 31) & ~31
        f.write(b'\0' * (aligned_end - f.tell()))

    file_size = os.path.getsize(output_path)
    print(f"\nGGUF file created: {output_path}")
    print(f"Size: {file_size / 1024 / 1024:.1f} MB")
    print(f"Tensors: {len(real_tensors)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default=r'models/gemma-4-E2B')
    parser.add_argument('--quantized-pt', default=r'quantized/gemma-4-E2B-sub1bit.pt')
    parser.add_argument('--output', default=r'quantized/gemma-4-E2B-sub1bit.gguf')
    args = parser.parse_args()
    build_gemma_gguf(args.model_dir, args.quantized_pt, args.output)