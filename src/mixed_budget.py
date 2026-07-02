from __future__ import annotations

from copy import deepcopy
from math import sqrt
from typing import Iterable, Sequence


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


def _normalise_mi_scores(mi_scores: Sequence[float], n_layers: int) -> list[float]:
    """Map raw MI scores to [0, 1] weights for the allocator.

    The allocator multiplies each layer's error-reduction score by
    (1 + mi_prior * mi_weight), so a weight of 1.0 means "fully MI-
    prioritised" and 0.0 means "ignore MI entirely".
    """
    if len(mi_scores) != n_layers:
        raise ValueError(
            f"mi_scores length {len(mi_scores)} != number of layers {n_layers}"
        )
    scores = [float(s) for s in mi_scores]
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [0.0] * n_layers
    return [(s - lo) / (hi - lo) for s in scores]


def allocate_mixed_budget(
    layers: list[dict],
    target_avg_bpw: float,
    mi_scores: Sequence[float] | None = None,
    mi_prior: float = 0.0,
) -> dict:
    """Allocate a bit budget across candidate formats.

    Parameters
    ----------
    layers : list[dict]
        Each entry has ``params``, ``candidates`` (list of dicts with
        ``bpw``, ``mse``, ``method``), and optionally ``activation_weight``.
    target_avg_bpw : float
        Average bits-per-weight across the whole model.
    mi_scores : sequence of float, optional
        Per-layer mutual-information scores from
        :class:`cross_layer_mi.MIAllocation`. When supplied, each
        layer's effective error-reduction is multiplied by
        ``(1 + mi_prior * normalised_mi_score)``. Higher MI -> the
        allocator prefers to upgrade that layer first when its error
        reduction is comparable.
    mi_prior : float
        Strength of the MI prior. 0.0 = pure MSE-driven allocation
        (the default); 1.0 = MI score and MSE contribute equally to
        the upgrade priority; values > 1.0 let MI dominate.

    Returns
    -------
    dict
        Allocation summary including the selected layer entries,
        average BPW, weighted MSE, and a method-count breakdown.
    """
    if target_avg_bpw <= 0:
        raise ValueError("target_avg_bpw must be positive")
    if not layers:
        raise ValueError("layers must not be empty")
    if mi_prior < 0:
        raise ValueError("mi_prior must be non-negative")

    candidate_layers: list[list[dict]] = []
    for layer in layers:
        candidates = [_candidate_score(layer, item) for item in layer.get("candidates", [])]
        if not candidates:
            raise ValueError(f"layer {layer.get('key', layer.get('idx'))} has no candidates")
        candidates.sort(key=lambda item: (item["bpw"], item["weighted_mse"], item["method"]))
        candidate_layers.append(candidates)

    mi_weights = (
        _normalise_mi_scores(mi_scores, len(layers)) if mi_scores else [0.0] * len(layers)
    )

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
                mi_multiplier = 1.0 + mi_prior * mi_weights[layer_idx]
                score = (error_reduction * mi_multiplier) / extra_bits
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
            "mi_prior": float(mi_prior),
            "mi_used": mi_scores is not None,
        }
    )
    return summary


def _method_counts(selected_layers: Iterable[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in selected_layers:
        method = str(item["method"])
        counts[method] = counts.get(method, 0) + 1
    return dict(sorted(counts.items()))
