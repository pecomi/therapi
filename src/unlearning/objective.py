"""Unlearning objective and matching evaluation helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_aligner(models, target_gex, source_gex):
    source_ae, target_encoder, emb_classifier, exp_classifier = models
    source_z = source_ae.encoder(source_gex)
    weights, latent, wgex, recon = target_encoder(target_gex, source_z, source_gex)
    return {
        "weights": weights,
        "latent": latent,
        "wgex": wgex,
        "recon": recon,
        "emb_logits": emb_classifier(latent),
        "exp_logits": exp_classifier(wgex),
    }


def alignment_losses(
    output,
    target_gex,
    labels,
    center_criterion,
    recon_weight,
    class_weight,
    center_weight,
):
    """Original THERAPI target loss, with its individual components."""
    recon = F.mse_loss(output["recon"], target_gex)
    emb_class = F.cross_entropy(output["emb_logits"], labels)
    exp_class = F.cross_entropy(output["exp_logits"], labels)
    center = center_criterion(output["latent"], labels)
    total = (
        recon_weight * recon
        + class_weight * (emb_class + exp_class)
        + center_weight * center
    )
    return {
        "task": total,
        "recon": recon,
        "emb_class": emb_class,
        "exp_class": exp_class,
        "center": center,
    }


def per_sample_alignment_losses(
    output,
    target_gex,
    labels,
    center_criterion,
    recon_weight,
    class_weight,
    center_weight,
):
    """Per-sample form of the same loss, used for exact group means."""
    recon = (output["recon"] - target_gex).pow(2).mean(dim=1)
    emb_class = F.cross_entropy(output["emb_logits"], labels, reduction="none")
    exp_class = F.cross_entropy(output["exp_logits"], labels, reduction="none")
    center = (output["latent"] - center_criterion.centers[labels]).pow(2).sum(dim=1)
    task = (
        recon_weight * recon
        + class_weight * (emb_class + exp_class)
        + center_weight * center
    )
    return {
        "task": task,
        "recon": recon,
        "emb_class": emb_class,
        "exp_class": exp_class,
        "center": center,
    }


@torch.no_grad()
def evaluate_loader(
    loader,
    models,
    center_criterion,
    source_gex,
    recon_weight,
    class_weight,
    center_weight,
):
    modules = (*models, center_criterion)
    modes = [module.training for module in modules]
    for module in modules:
        module.eval()

    sums = {name: 0.0 for name in ("task", "recon", "emb_class", "exp_class", "center")}
    correct_emb = correct_exp = count = 0
    for target_gex, _, labels in loader:
        target_gex = target_gex.to(source_gex.device)
        labels = labels.to(source_gex.device)
        output = forward_aligner(models, target_gex, source_gex)
        losses = per_sample_alignment_losses(
            output,
            target_gex,
            labels,
            center_criterion,
            recon_weight,
            class_weight,
            center_weight,
        )
        for name, values in losses.items():
            sums[name] += values.sum().item()
        correct_emb += (output["emb_logits"].argmax(dim=1) == labels).sum().item()
        correct_exp += (output["exp_logits"].argmax(dim=1) == labels).sum().item()
        count += len(target_gex)

    for module, was_training in zip(modules, modes):
        module.train(was_training)
    return {
        **{name: value / count for name, value in sums.items()},
        "emb_accuracy": correct_emb / count,
        "exp_accuracy": correct_exp / count,
        "n_samples": count,
    }
