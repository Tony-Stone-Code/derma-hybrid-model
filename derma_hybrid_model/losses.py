from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MulticlassFocalLoss(nn.Module):
    def __init__(
        self,
        *,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()

        log_pt = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.alpha is None:
            alpha_factor = 1.0
        else:
            alpha = self.alpha.to(logits.device)
            alpha_factor = alpha.gather(0, targets)

        loss = -alpha_factor * focal_factor * log_pt
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
