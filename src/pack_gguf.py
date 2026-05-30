import torch
import os
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from .gguf_writer import GGUFWriter, GGML_TYPES


@dataclass
class QuantizedLayer:
    U: torch.Tensor
    S: torch.Tensor
    Vt: torch.Tensor
    U_scale: torch.Tensor
    S_scale: torch.Tensor
    bit_allocations: Tuple[int, int, int]


def create_lowrank_type_defs() -> str:
    return """
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q8_0    = 2,
    GGML_TYPE_Q4_0    = 3,
    GGML_TYPE_Q4_1    = 4,
    GGML_TYPE_LOWRANK_UV_0BIT  = 100,
    GGML_TYPE_LOWRANK_UV_1BIT  = 101,
    GGML_TYPE_LOWRANK_SIGMA_2BIT = 102,
};
"""


def pack_sub1bit_model(
    factors: Dict[int, Dict],
    output_path: str,
    model_name: str = "llama-2-7b-sub1bit",
    metadata: Optional[Dict] = None
):
    writer = GGUFWriter(output_path)
    writer.add_key_value("general.architecture", "llama")
    writer.add_key_value("general.name", model_name)
    writer.add_key_value("general.file_type", "sub1bit")
    if metadata:
        for key, value in metadata.items():
            writer.add_key_value(key, value)
    for layer_idx, layer_data in factors.items():
        writer.add_tensor(
            f"layer.{layer_idx}.U",
            layer_data.get('U_packed', layer_data['U']).cpu().numpy(),
            GGML_TYPES['int8']
        )
        writer.add_tensor(
            f"layer.{layer_idx}.S",
            layer_data['S'].cpu().numpy().astype(np.float16),
            GGML_TYPES['float16']
        )
        writer.add_tensor(
            f"layer.{layer_idx}.Vt",
            layer_data.get('Vt_packed', layer_data['Vt']).cpu().numpy(),
            GGML_TYPES['int8']
        )
        if 'U_scale' in layer_data:
            writer.add_tensor(
                f"layer.{layer_idx}.U_scale",
                np.array([layer_data['U_scale'].item()], dtype=np.float32),
                GGML_TYPES['float32']
            )
        if 'Vt_scale' in layer_data:
            writer.add_tensor(
                f"layer.{layer_idx}.Vt_scale",
                np.array([layer_data['Vt_scale'].item()], dtype=np.float32),
                GGML_TYPES['float32']
            )
        if 'S_scale' in layer_data:
            writer.add_tensor(
                f"layer.{layer_idx}.S_scale",
                np.array([layer_data['S_scale'].item()], dtype=np.float32),
                GGML_TYPES['float32']
            )
    writer.write()
    return os.path.getsize(output_path)


if __name__ == "__main__":
    dummy_factors = {
        0: {
            'U': torch.randn(4096, 16),
            'S': torch.randn(16),
            'Vt': torch.randn(16, 4096),
            'rank': 16
        }
    }
    output_path = "C:/Users/Zwmar/projects/sub1quant/quantized/test.gguf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    size = pack_sub1bit_model(dummy_factors, output_path)
    print(f"GGUF file created: {size} bytes")
