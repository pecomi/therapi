"""Small helpers for explicit, reproducible epoch-based LR schedules."""

from __future__ import annotations

import torch


def parse_lr_schedule(spec: str | None, *, total_epochs: int) -> dict[int, float]:
    """Parse ``EPOCH:LR`` pairs applied at the start of one-based epochs.

    Example: ``101:3e-4,161:1e-4`` keeps the optimizer's initial LR through
    epoch 100, switches to 3e-4 for epochs 101--160, and to 1e-4 thereafter.
    """
    if spec is None or not spec.strip():
        return {}

    schedule: dict[int, float] = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item or ":" not in item:
            raise ValueError(
                "LR schedule must be comma-separated EPOCH:LR pairs, "
                f"got {spec!r}"
            )
        raw_epoch, raw_lr = item.split(":", 1)
        try:
            epoch = int(raw_epoch)
            lr = float(raw_lr)
        except ValueError as error:
            raise ValueError(f"invalid LR schedule entry: {item!r}") from error
        if not 1 <= epoch <= total_epochs:
            raise ValueError(
                f"LR schedule epoch must be in [1, {total_epochs}], got {epoch}"
            )
        if lr <= 0:
            raise ValueError(f"scheduled LR must be positive, got {lr}")
        if epoch in schedule:
            raise ValueError(f"duplicate LR schedule epoch: {epoch}")
        schedule[epoch] = lr
    return dict(sorted(schedule.items()))


def set_optimizer_lr(optimizer, lr: float) -> None:
    """Set the LR of every parameter group in an optimizer."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def optimizer_lr(optimizer) -> float:
    """Return the common LR and reject unexpected mixed-group schedules."""
    values = {float(group["lr"]) for group in optimizer.param_groups}
    if len(values) != 1:
        raise ValueError(f"optimizer parameter groups have different LRs: {values}")
    return values.pop()


def create_lr_scheduler(
    optimizer,
    *,
    name: str,
    total_epochs: int,
    cosine_eta_min: float,
):
    """Create an optional epoch scheduler while preserving constant-LR defaults."""
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"unsupported LR scheduler: {name}")

    initial_lr = optimizer_lr(optimizer)
    if not 0 <= cosine_eta_min < initial_lr:
        raise ValueError(
            "cosine eta_min must be non-negative and smaller than the initial LR, "
            f"got eta_min={cosine_eta_min} initial_lr={initial_lr}"
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=cosine_eta_min,
    )
