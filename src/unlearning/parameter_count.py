"""Consistent parameter counts for THERAPI aligner training runs."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


def _count_unique(parameters: Iterable[torch.nn.Parameter]) -> int:
    """Count parameter elements once even if an iterable contains aliases."""
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        parameter_id = id(parameter)
        if parameter_id not in seen:
            seen.add(parameter_id)
            total += parameter.numel()
    return total


def aligner_parameter_counts(
    source_ae: nn.Module,
    target_encoder: nn.Module,
    emb_classifier: nn.Module,
    exp_classifier: nn.Module,
    center: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    """Return comparable counts for baseline, retrain, and unlearning runs.

    ``effective_output_path`` is the parameterized path that determines the
    deployed attention-weighted expression: source encoder plus target Q/K.
    Decoders, classifiers, and center anchors do not directly determine that
    output, although the classifiers and centers provide training gradients.
    """
    components = {
        "source_encoder": _count_unique(source_ae.encoder.parameters()),
        "source_decoder": _count_unique(source_ae.decoder.parameters()),
        "target_Q": _count_unique(target_encoder.Q.parameters()),
        "target_K": _count_unique(target_encoder.K.parameters()),
        "target_decoder": _count_unique(target_encoder.decoder.parameters()),
        "latent_classifier": _count_unique(emb_classifier.parameters()),
        "expression_classifier": _count_unique(exp_classifier.parameters()),
        "center": _count_unique(center.parameters()),
    }
    optimizer_parameters = (
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    counts = {
        **components,
        "aligner_modules_total": (
            components["source_encoder"]
            + components["source_decoder"]
            + components["target_Q"]
            + components["target_K"]
            + components["target_decoder"]
        ),
        "effective_output_path": (
            components["source_encoder"]
            + components["target_Q"]
            + components["target_K"]
        ),
        "checkpoint_total": sum(components.values()),
        "optimizer_updated": _count_unique(optimizer_parameters),
    }
    counts["optimizer_fixed"] = (
        counts["checkpoint_total"] - counts["optimizer_updated"]
    )
    return counts


def format_parameter_counts(counts: dict[str, int]) -> str:
    """Format the headline counts as one grep-friendly training-log row."""
    keys = (
        "checkpoint_total",
        "aligner_modules_total",
        "effective_output_path",
        "optimizer_updated",
        "optimizer_fixed",
    )
    values = " ".join(f"{key}={counts[key]}" for key in keys)
    return f"[parameters] {values}"
