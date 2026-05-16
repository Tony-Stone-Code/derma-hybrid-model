# derma-hybrid-model

PyTorch utilities for a leak-proof, explainable 7-class HAM10000 classifier built around a 4-stream feature-fusion architecture.

## Included components

- Lesion-level `StratifiedGroupKFold` train/validation/test splitting to prevent HAM10000 leakage.
- Albumentations pipelines with heavier minority-class augmentation.
- A fused classifier that combines MobileNetV2, EfficientNet-B2, ResNet50, and DenseNet121 features.
- Multi-class focal loss with inverse-frequency class weights.
- Mixed-precision, gradient-accumulation, checkpointing, and optional feature caching helpers for Kaggle.
- Integrated Gradients utilities for full-model attributions and overlay rendering.

## Install

```bash
pip install -r requirements.txt
```

## Example

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader

from derma_hybrid_model import (
    DermoscopyDataset,
    HybridDermClassifier,
    MulticlassFocalLoss,
    build_transforms,
    compute_class_weights,
    create_train_val_test_split,
    evaluate_model,
    train_one_epoch,
)

metadata = pd.read_csv("HAM10000_metadata.csv")
splits = create_train_val_test_split(metadata)
base_transform, minority_transform = build_transforms(image_size=224)

train_dataset = DermoscopyDataset(
    splits["train"],
    image_root="HAM10000_images",
    transform=base_transform,
    minority_transform=minority_transform,
)
val_dataset = DermoscopyDataset(
    splits["val"],
    image_root="HAM10000_images",
    transform=base_transform,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridDermClassifier().to(device)
class_weights = compute_class_weights(splits["train"]).to(device)
criterion = MulticlassFocalLoss(alpha=class_weights, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

loss = train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device=device,
    accumulation_steps=4,
    use_amp=True,
)
metrics = evaluate_model(model, val_loader, device=device)
print(loss, metrics["macro_f1"], metrics["balanced_accuracy"])
```
