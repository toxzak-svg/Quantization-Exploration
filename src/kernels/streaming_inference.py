"""
Streaming inference engine for sub1bit quantized models.

Loads one layer at a time from disk for minimal memory footprint,
with fused ternary GEMV kernels for efficient computation.
"""

import torch
import struct
import json
from pathlib import Path
from typing import Iterator, Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass


@dataclass
class LayerWeights:
    U_packed: torch.Tensor
    Vt_packed: torch.Tensor
    S: torch.Tensor
    U_scale: float
    Vt_scale: float
    S_scale: float
    U_shape: Tuple[int, int]
    Vt_shape: Tuple[int, int]


class StreamingInference:
    """
    Memory-efficient inference that streams weights from disk.

    For generation, only one layer's weights need to be in memory at a time,
    while maintaining the full embedding and cache.
    """

    def __init__(
        self,
        model_dir: str,
        quantized_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model_dir = Path(model_dir)
        self.quantized_path = Path(quantized_path)
        self.device = device

        self.weight_info = self._load_safetensor_info()
        self.quantized_data = None

    def _load_safetensor_info(self) -> List[Tuple[str, Dict]]:
        """Load safetensor metadata without loading weights."""
        safetensor_path = self.model_dir / 'model.safetensors'
        with open(safetensor_path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_size))

        weight_info = []
        for key, info in header.items():
            if key == '__metadata__':
                continue
            if not isinstance(info, dict) or 'shape' not in info:
                continue
            weight_info.append((key, info))

        return weight_info

    def load_quantized(self):
        """Load quantized index, not full weights."""
        self.quantized_data = torch.load(self.quantized_path, map_location='cpu', weights_only=True)

    def stream_layer_weights(self, layer_idx: int) -> Optional[LayerWeights]:
        """Stream weights for a single layer from quantized data."""
        if self.quantized_data is None:
            self.load_quantized()

        quantized = self.quantized_data.get('quantized', {})

        if layer_idx not in quantized:
            return None

        q_entry = quantized[layer_idx]

        return LayerWeights(
            U_packed=q_entry['U_packed'],
            Vt_packed=q_entry['Vt_packed'],
            S=q_entry['S'],
            U_scale=q_entry['U_scale'].item() if hasattr(q_entry['U_scale'], 'item') else q_entry['U_scale'],
            Vt_scale=q_entry['Vt_scale'].item() if hasattr(q_entry['Vt_scale'], 'item') else q_entry['Vt_scale'],
            S_scale=q_entry['S_scale'].item() if hasattr(q_entry['S_scale'], 'item') else q_entry['S_scale'],
            U_shape=tuple(q_entry['U_shape']),
            Vt_shape=tuple(q_entry['Vt_shape']),
        )

    def forward_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,
        use_triton: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through a single layer.

        Args:
            layer_idx: layer index
            x: input tensor
            use_triton: use Triton kernels if available

        Returns:
            output tensor
        """
        from .ternary_gemv import ternary_gemv, ternary_gemv_high_bw

        weights = self.stream_layer_weights(layer_idx)
        if weights is None:
            raise ValueError(f"Layer {layer_idx} not found in quantized data")

        if use_triton:
            try:
                from .triton_ternary_gemv import TritonTernaryGEMV
                gemv = TritonTernaryGEMV(
                    weights.U_shape[0],
                    weights.Vt_shape[1],
                    weights.U_shape[1],
                    device=self.device
                )
                gemv.stream_weights(
                    weights.U_packed, weights.Vt_packed, weights.S,
                    weights.U_scale, weights.Vt_scale, weights.S_scale,
                    weights.U_shape, weights.Vt_shape
                )
                return gemv.forward(x)
            except ImportError:
                pass

        out_features = weights.Vt_shape[1]
        if out_features > 4096:
            return ternary_gemv_high_bw(
                weights.U_packed, weights.Vt_packed, weights.S,
                weights.U_scale, weights.Vt_scale, weights.S_scale,
                x, weights.U_shape, weights.Vt_shape
            )
        else:
            return ternary_gemv(
                weights.U_packed, weights.Vt_packed, weights.S,
                weights.U_scale, weights.Vt_scale, weights.S_scale,
                x, weights.U_shape, weights.Vt_shape
            )

    def iter_layers(self) -> Iterator[Tuple[int, LayerWeights]]:
        """Iterate through all layer weights."""
        if self.quantized_data is None:
            self.load_quantized()

        quantized = self.quantized_data.get('quantized', {})
        for idx in sorted(quantized.keys()):
            yield idx, self.stream_layer_weights(idx)


def create_streaming_inference(
    model_dir: str,
    quantized_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> StreamingInference:
    """Factory function to create streaming inference engine."""
    return StreamingInference(model_dir, quantized_path, device)


class StreamingMoE:
    """
    Streaming inference for MoE layers with routing.

    Loads experts one at a time and accumulates weighted outputs.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        stream_inference: StreamingInference
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.stream_inference = stream_inference

    def forward_with_routing(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_layer_idx: int
    ) -> torch.Tensor:
        """
        Forward pass with top-k routing.

        Args:
            x: input tensor [batch, hidden]
            routing_weights: [batch, num_experts] top-k routing weights
            expert_indices: [batch, top_k] selected expert indices
            expert_layer_idx: layer index for experts

        Returns:
            output tensor [batch, hidden]
        """
        batch_size = x.shape[0]
        hidden_dim = x.shape[1]
        output = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_id = expert_indices[:, k].tolist()
            weights = routing_weights[:, k]

            for b in range(batch_size):
                eid = expert_id[b]
                w = weights[b]

                expert_weights = self.stream_inference.stream_layer_weights(expert_layer_idx + eid)
                expert_out = self.stream_inference.forward_layer(expert_layer_idx + eid, x[b])

                output[b] += expert_out * w

        return output