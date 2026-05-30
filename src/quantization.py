import torch
import numpy as np


def ternary_quantize(x: torch.Tensor):
    scale = x.abs().max()
    if scale == 0:
        scale = 1.0
    normalized = x / scale
    ternary = torch.sign(normalized)
    ternary[ternary == 0] = 1
    return ternary.to(torch.int8), scale


def ternary_pack(t: torch.Tensor) -> torch.Tensor:
    encoded = t.to(torch.int8) + 1
    n = encoded.numel()
    pad = (5 - n % 5) % 5
    if pad:
        encoded = torch.cat([encoded.flatten(), torch.zeros(pad, dtype=torch.int8, device=encoded.device)])
    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=encoded.device)
    packed = (encoded.reshape(-1, 5).to(torch.int32) * weights).sum(dim=1)
    return packed.to(torch.uint8)


def ternary_unpack(packed: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=packed.device)
    expanded = packed.to(torch.int32).unsqueeze(-1) // weights % 3
    flat = (expanded - 1).flatten()
    return flat[:np.prod(original_shape)].reshape(original_shape).to(torch.int8)


def sigma_quantize(s: torch.Tensor, num_bits: int = 2):
    max_val = s.abs().max()
    if max_val == 0:
        max_val = 1.0
    qmax = 2 ** (num_bits - 1) - 1
    scale = max_val / qmax
    quantized = (s / scale).round().clamp(-qmax, qmax)
    return quantized.to(torch.int8), scale


def sigma_dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def quantize_factor(U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, sigma_bits: int = 2):
    U_q, U_scale = ternary_quantize(U)
    Vt_q, Vt_scale = ternary_quantize(Vt)
    S_q, S_scale = sigma_quantize(S, sigma_bits)
    return {
        'U': U_q, 'U_scale': U_scale,
        'Vt': Vt_q, 'Vt_scale': Vt_scale,
        'S': S_q, 'S_scale': S_scale,
    }


def pack_factor(data: dict):
    return {
        'U_packed': ternary_pack(data['U']),
        'U_scale': data['U_scale'],
        'Vt_packed': ternary_pack(data['Vt']),
        'Vt_scale': data['Vt_scale'],
        'S': data['S'],
        'S_scale': data['S_scale'],
        'U_shape': data['U'].shape,
        'Vt_shape': data['Vt'].shape,
    }


def unpack_factor(data: dict):
    if 'U_packed' in data:
        U = ternary_unpack(data['U_packed'], data['U_shape'])
        Vt = ternary_unpack(data['Vt_packed'], data['Vt_shape'])
    else:
        U = data['U']
        Vt = data['Vt']
    return {
        'U': U,
        'U_scale': data['U_scale'],
        'Vt': Vt,
        'Vt_scale': data['Vt_scale'],
        'S': data['S'],
        'S_scale': data['S_scale'],
    }


def dequantize_factor(data: dict) -> torch.Tensor:
    if 'U_packed' in data:
        U = ternary_unpack(data['U_packed'], data['U_shape']).float()
        Vt = ternary_unpack(data['Vt_packed'], data['Vt_shape']).float()
    else:
        U = data['U'].float()
        Vt = data['Vt'].float()
    S = sigma_dequantize(data['S'], data['S_scale'])
    return torch.matmul(U * S.unsqueeze(0), Vt)
