"""Small helpers for explicit, reproducible epoch-based LR schedules."""

from __future__ import annotations


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
