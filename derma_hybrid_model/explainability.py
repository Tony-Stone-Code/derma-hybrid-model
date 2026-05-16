from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_uint8_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
    else:
        array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {array.shape}")
    if array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def summarize_attributions(attributions: torch.Tensor) -> np.ndarray:
    heatmap = attributions.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()
    heatmap -= heatmap.min()
    denominator = heatmap.max()
    if denominator > 0:
        heatmap /= denominator
    return heatmap


def overlay_attribution(
    image: torch.Tensor | np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    image_array = _to_uint8_image(image).astype(np.float32)
    heat = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    heat_rgb = np.stack([heat * 255.0, np.zeros_like(heat), (1.0 - heat) * 255.0], axis=-1)
    overlay = image_array * (1.0 - alpha) + heat_rgb * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def generate_integrated_gradients(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    target: int | None = None,
    baseline: torch.Tensor | None = None,
    steps: int = 100,
    integrated_gradients_cls: Any | None = None,
) -> dict[str, Any]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if baseline is None:
        baseline = torch.zeros_like(image)

    model.eval()
    with torch.no_grad():
        logits = model(image)
        resolved_target = int(logits.argmax(dim=1).item()) if target is None else int(target)

    if integrated_gradients_cls is None:
        try:
            from captum.attr import IntegratedGradients
        except ImportError as exc:
            raise ImportError("Captum is required to generate Integrated Gradients.") from exc
        integrated_gradients_cls = IntegratedGradients

    integrated_gradients = integrated_gradients_cls(model)
    attributions, convergence_delta = integrated_gradients.attribute(
        image,
        baselines=baseline,
        target=resolved_target,
        n_steps=steps,
        return_convergence_delta=True,
    )
    return {
        "target": resolved_target,
        "attributions": attributions.detach(),
        "delta": convergence_delta.detach()
        if hasattr(convergence_delta, "detach")
        else convergence_delta,
    }


def save_explanation_triptych(
    image: torch.Tensor | np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save explanation figures.") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_to_uint8_image(image))
    axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="inferno")
    axes[1].set_title("IG Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output
