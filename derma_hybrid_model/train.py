from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .data import DEFAULT_CLASS_NAMES
from .evaluation import compute_classification_metrics


def _resolve_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: str | torch.device,
    accumulation_steps: int = 1,
    use_amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    resolved_device = _resolve_device(device)
    use_cuda_amp = use_amp and resolved_device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    batch_count = 0

    for step, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(resolved_device)
        targets = targets.to(resolved_device)
        with torch.amp.autocast(device_type=resolved_device.type, enabled=use_cuda_amp):
            logits = model(images)
            loss = criterion(logits, targets) / accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % accumulation_steps == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item() * accumulation_steps)
        batch_count += 1

    if batch_count % accumulation_steps != 0:
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(batch_count, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: Iterable,
    *,
    device: str | torch.device,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> dict[str, object]:
    resolved_device = _resolve_device(device)
    model.eval()
    probabilities = []
    targets = []

    for images, batch_targets in dataloader:
        logits = model(images.to(resolved_device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.extend(batch_targets.numpy().tolist())

    probability_matrix = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, len(class_names)))
    return compute_classification_metrics(targets, probability_matrix, class_names)


@torch.no_grad()
def cache_fused_features(
    model: nn.Module,
    dataloader: Iterable,
    output_path: str | Path,
    *,
    device: str | torch.device,
) -> Path:
    resolved_device = _resolve_device(device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    cached_features = []
    cached_targets = []
    for images, targets in dataloader:
        fused = model.extract_features(images.to(resolved_device)).cpu()
        cached_features.append(fused)
        cached_targets.append(targets.cpu())

    payload = {
        "features": torch.cat(cached_features, dim=0) if cached_features else torch.empty(0),
        "targets": torch.cat(cached_targets, dim=0) if cached_targets else torch.empty(0, dtype=torch.long),
    }
    torch.save(payload, output)
    return output


def save_checkpoint(
    checkpoint_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler | None = None,
    extra_state: dict[str, object] | None = None,
) -> Path:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"epoch-{epoch:03d}.pt"
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, checkpoint_path)
    return checkpoint_path
