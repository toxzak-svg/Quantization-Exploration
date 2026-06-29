from __future__ import annotations

from copy import deepcopy
from math import sqrt
from typing import Iterable


def _candidate_score(layer: dict, candidate: dict) -> dict:
    params = int(layer["params"])
    activation_weight = float(layer.get("activation_weight", 1.0))
    scored = deepcopy(candidate)
    scored["idx"] = layer.get("idx")
    scored["key"] = layer.get("key")
    scored["params"] = params
    scored["activation_weight"] = activation_weight
    scored["weighted_mse"] = float(candidate["mse"]) * activation_weight
    scored["total_bits"] = float(candidate["bpw"]) * params
    scored["weighted_sse"] = scored["weighted_mse"] * params
    return scored


def summarize_allocation(selected_layers: Iterable[dict]) -> dict:
    selected = list(selected_layers)
    total_params = sum(int(item["params"]) for item in selected)
    if total_params <= 0:
        raise ValueError("allocation must contain at least one parameter")

    total_bits = sum(float(item["bpw"]) * int(item["params"]) for item in selected)
    weighted_sse = sum(float(item.get("weighted_mse", item["mse"])) * int(item["params"]) for item in selected)
    weighted_mse = weighted_sse / total_params
    return {
        "layers": len(selected),
        "total_params": total_params,
        "avg_bpw": total_bits / total_params,
        "weighted_mse": weighted_mse,
        "weighted_rmse": sqrt(weighted_mse),
        "compression_vs_bf16": 16.0 / (total_bits / total_params),
    }


def allocate_mixed_budget(layers: list[dict], target_avg_bpw: float) -> dict:
    if target_avg_bpw <= 0:
        raise ValueError("target_avg_bpw must be positive")
    if not layers:
        raise ValueError("layers must not be empty")

    candidate_layers: list[list[dict]] = []
    for layer in layers:
        candidates = [_candidate_score(layer, item) for item in layer.get("candidates", [])]
        if not candidates:
            raise ValueError(f"layer {layer.get('key', layer.get('idx'))} has no candidates")
        candidates.sort(key=lambda item: (item["bpw"], item["weighted_mse"], item["method"]))
        candidate_layers.append(candidates)

    selected_indices = [0 for _ in candidate_layers]
    selected = [candidates[0] for candidates in candidate_layers]
    total_params = sum(item["params"] for item in selected)
    budget_bits = target_avg_bpw * total_params
    current_bits = sum(item["total_bits"] for item in selected)
    if current_bits > budget_bits + 1e-9:
        raise ValueError("cheapest candidates exceed target_avg_bpw")

    while True:
        best_upgrade = None
        for layer_idx, candidates in enumerate(candidate_layers):
            current = selected[layer_idx]
            for candidate_idx, candidate in enumerate(candidates):
                if candidate_idx == selected_indices[layer_idx]:
                    continue
                extra_bits = candidate["total_bits"] - current["total_bits"]
                if extra_bits <= 0:
                    continue
                if current_bits + extra_bits > budget_bits + 1e-9:
                    continue
                error_reduction = current["weighted_sse"] - candidate["weighted_sse"]
                if error_reduction <= 0:
                    continue
                score = error_reduction / extra_bits
                contender = (score, error_reduction, -extra_bits, layer_idx, candidate_idx, candidate)
                if best_upgrade is None or contender > best_upgrade:
                    best_upgrade = contender

        if best_upgrade is None:
            break

        _, _, _, layer_idx, candidate_idx, candidate = best_upgrade
        current_bits += candidate["total_bits"] - selected[layer_idx]["total_bits"]
        selected_indices[layer_idx] = candidate_idx
        selected[layer_idx] = candidate

    summary = summarize_allocation(selected)
    summary.update(
        {
            "target_avg_bpw": float(target_avg_bpw),
            "selected_layers": selected,
            "method_counts": _method_counts(selected),
        }
    )
    return summary


def _method_counts(selected_layers: Iterable[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in selected_layers:
        method = str(item["method"])
        counts[method] = counts.get(method, 0) + 1
    return dict(sorted(counts.items()))
