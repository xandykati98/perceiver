#!/usr/bin/env python3
"""
LAMB optimizer (Layer-wise Adaptive Moments) for PyTorch.

Based on: "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song, James Demmel, and Cho-Jui Hsieh.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    """
    LAMB optimizer.

    Notes:
    - This implementation follows the common formulation where the trust ratio is:
        trust_ratio = ||w|| / ||update||
      and the parameter update is:
        w <- w - lr * trust_ratio * update
    - Weight decay is applied in the update term (AdamW-style, but inside the update vector).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        bias_correction: bool,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "bias_correction": bias_correction,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps: float = float(group["eps"])
            weight_decay: float = float(group["weight_decay"])
            bias_correction: bool = bool(group["bias_correction"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                state["step"] += 1
                step: int = int(state["step"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if bias_correction:
                    bias_correction1 = 1.0 - (beta1**step)
                    bias_correction2 = 1.0 - (beta2**step)
                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2
                else:
                    m_hat = exp_avg
                    v_hat = exp_avg_sq

                update = m_hat / (torch.sqrt(v_hat) + eps)
                if weight_decay != 0.0:
                    update = update.add(p, alpha=weight_decay)

                w_norm = torch.norm(p)
                u_norm = torch.norm(update)
                if w_norm > 0.0 and u_norm > 0.0:
                    trust_ratio = (w_norm / u_norm).clamp(min=0.0)
                else:
                    trust_ratio = torch.ones_like(w_norm)

                p.add_(update, alpha=-lr * trust_ratio.item())

        return loss


