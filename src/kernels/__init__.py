"""Custom kernels for sub1bit streaming inference."""

from .ternary_gemv import ternary_gemv, ternary_gemv_high_bw, _unpack_ternary
from .streaming_inference import (
    StreamingInference,
    StreamingMoE,
    LayerWeights,
    create_streaming_inference
)

try:
    from .triton_ternary_gemv import TritonTernaryGEMV, triton_unpack_ternary
    __all__ = [
        'ternary_gemv', 'ternary_gemv_high_bw', '_unpack_ternary',
        'TritonTernaryGEMV', 'triton_unpack_ternary',
        'StreamingInference', 'StreamingMoE', 'LayerWeights',
        'create_streaming_inference'
    ]
except ImportError:
    __all__ = [
        'ternary_gemv', 'ternary_gemv_high_bw', '_unpack_ternary',
        'StreamingInference', 'StreamingMoE', 'LayerWeights',
        'create_streaming_inference'
    ]