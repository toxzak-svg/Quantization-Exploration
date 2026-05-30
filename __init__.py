import torch
from typing import Dict, Optional, List
from .Sub1BitLLM import Sub1BitLLM, Sub1BitConfig, from_fp16
from .train_transform import BTCLLMTrainer, LearnableTransform, BinaryCodebook, TransformLayer
from .lowrank_factorization import low_rank_factorize, factorize_model_weights, compute_optimal_rank
from .pack_gguf import pack_sub1bit_model
from .gguf_writer import GGUFWriter, GGML_TYPES, GGUF_TYPES
from .quantization import (
    ternary_quantize, ternary_pack, ternary_unpack,
    sigma_quantize, sigma_dequantize,
    quantize_factor, pack_factor, unpack_factor, dequantize_factor,
)

__all__ = [
    "Sub1BitLLM",
    "Sub1BitConfig",
    "from_fp16",
    "BTCLLMTrainer",
    "LearnableTransform",
    "BinaryCodebook",
    "TransformLayer",
    "low_rank_factorize",
    "factorize_model_weights",
    "compute_optimal_rank",
    "pack_sub1bit_model",
    "GGUFWriter",
    "GGML_TYPES",
    "GGUF_TYPES",
    "ternary_quantize",
    "ternary_pack",
    "ternary_unpack",
    "sigma_quantize",
    "sigma_dequantize",
    "quantize_factor",
    "pack_factor",
    "unpack_factor",
    "dequantize_factor",
]


class Sub1BitHooks:
    @staticmethod
    def pre_transform_hook(layer_idx: int, weight: torch.Tensor) -> torch.Tensor:
        return weight

    @staticmethod
    def post_transform_hook(layer_idx: int, transformed: torch.Tensor) -> torch.Tensor:
        return transformed

    @staticmethod
    def pre_quantization_hook(layer_idx: int, factors: dict) -> dict:
        return factors

    @staticmethod
    def post_reconstruction_hook(layer_idx: int, reconstructed: torch.Tensor) -> torch.Tensor:
        return reconstructed


class Sub1BitPipeline:
    def __init__(self, config: Sub1BitConfig):
        self.config = config
        self.hooks = Sub1BitHooks()
        self._trainer = None
        self._quantized_model = None

    def train_transforms(self, weight_data: Dict[int, torch.Tensor], epochs: int = 5, **kwargs):
        layer_sizes = [(w.shape[0], w.shape[1]) for w in weight_data.values()]
        self._trainer = BTCLLMTrainer(
            layer_sizes=layer_sizes,
            codebook_dim=self.config.codebook_dim,
            **kwargs
        )
        for epoch in range(epochs):
            loss = self._trainer.train_epoch(weight_data)
            print(f"Transform epoch {epoch+1}/{epochs}, loss={loss:.6f}")
        return self

    def build_codebook(self, checkpoint_path: str):
        self._trainer.save(checkpoint_path)
        return self

    def factorize_and_quantize(self, weight_data: Dict[int, torch.Tensor], output_path: str):
        from .quantize import SubOneBitQuantizer
        quantizer = SubOneBitQuantizer(model_path="dummy")
        quantized = {}
        for idx, W in weight_data.items():
            factors = low_rank_factorize(W.float(), self.config.energy_threshold)
            factors = self.hooks.pre_quantization_hook(idx, factors)
            quantized[idx] = factors
        return quantized

    def export(self, output_path: str, metadata: Optional[Dict] = None):
        if self._quantized_model is None:
            raise ValueError("No quantized model to export. Run factorize_and_quantize first.")
        self._quantized_model.to_gguf(output_path, metadata)
        return self


__all__ += ["Sub1BitHooks", "Sub1BitPipeline"]


if __name__ == "__main__":
    import torch
    config = Sub1BitConfig(model_name="test")
    print("Sub1BitLLM API loaded successfully")
    print(f"Available exports: {__all__}")
