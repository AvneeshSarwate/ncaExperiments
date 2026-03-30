"""Rollout functions for NCA experiments."""

import torch

from .nca_model import NCAModel


def _ensure_cond_tensor(cond_ids, batch_size, device):
    """Convert cond_ids to a [B] long tensor if it's a scalar int."""
    if cond_ids is None:
        return None
    if isinstance(cond_ids, int):
        return torch.full((batch_size,), cond_ids, dtype=torch.long, device=device)
    return cond_ids


def rollout_fixed(model, x, steps, cond_ids=None):
    """Run model for N steps, return final state.

    Args:
        model: NCAModel instance
        x: [B, C, H, W] initial state tensor
        steps: number of NCA steps to run
        cond_ids: optional int or [B] int tensor of class indices
    Returns:
        [B, C, H, W] final state tensor
    """
    cond_ids = _ensure_cond_tensor(cond_ids, x.shape[0], x.device)
    for _ in range(steps):
        x = model(x, cond_ids=cond_ids)
    return x


def rollout_switch(model, x, steps_a, cond_a, steps_b, cond_b,
                   reset_hidden=True):
    """Run model for steps_a with cond_a, then steps_b with cond_b.

    Optionally resets hidden channels between phases (zeros channels 4:,
    preserves 0:4).

    Args:
        model: NCAModel instance
        x: [B, C, H, W] initial state tensor
        steps_a: number of steps for first phase
        cond_a: [B] int tensor of class indices for phase A
        steps_b: number of steps for second phase
        cond_b: [B] int tensor of class indices for phase B
        reset_hidden: if True, zero hidden channels between phases
    Returns:
        [B, C, H, W] final state tensor
    """
    x = rollout_fixed(model, x, steps_a, cond_ids=cond_a)

    if reset_hidden:
        x = NCAModel.reset_hidden(x)

    x = rollout_fixed(model, x, steps_b, cond_ids=cond_b)
    return x
