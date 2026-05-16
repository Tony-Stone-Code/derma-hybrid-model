from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "nv",
    "mel",
    "bkl",
    "bcc",
    "akiec",
    "vasc",
    "df",
)


def _require_columns(metadata: pd.DataFrame, required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")


def create_train_val_test_split(
    metadata: pd.DataFrame,
    *,
    label_column: str = "dx",
    group_column: str = "lesion_id",
    n_splits: int = 5,
    val_fold: int = 0,
    test_fold: int = 1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create leak-proof train/val/test splits using StratifiedGroupKFold."""
    _require_columns(metadata, [label_column, group_column])
    if val_fold == test_fold:
        raise ValueError("val_fold and test_fold must be different")
    if n_splits < 3:
        raise ValueError("n_splits must be at least 3 to produce train/val/test partitions")
    if not 0 <= val_fold < n_splits or not 0 <= test_fold < n_splits:
        raise ValueError("fold indices must be within [0, n_splits)")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_assignments = np.empty(len(metadata), dtype=np.int64)

    for fold_index, (_, fold_indices) in enumerate(
        splitter.split(metadata, metadata[label_column], groups=metadata[group_column])
    ):
        fold_assignments[fold_indices] = fold_index

    split_frames = {
        "train": metadata.loc[~np.isin(fold_assignments, [val_fold, test_fold])].reset_index(drop=True),
        "val": metadata.loc[fold_assignments == val_fold].reset_index(drop=True),
        "test": metadata.loc[fold_assignments == test_fold].reset_index(drop=True),
    }
    return split_frames


def compute_class_weights(
    metadata: pd.DataFrame,
    *,
    label_column: str = "dx",
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> torch.Tensor:
    """Return inverse-frequency class weights normalized to mean 1."""
    _require_columns(metadata, [label_column])
    counts = metadata[label_column].value_counts()
    weights = []
    for class_name in class_names:
        count = int(counts.get(class_name, 0))
        if count <= 0:
            raise ValueError(f"class {class_name!r} is missing from metadata")
        weights.append(1.0 / count)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor / weights_tensor.mean()


def build_transforms(image_size: int = 224) -> tuple[object, object]:
    """Build base and minority-class augmentation pipelines with Albumentations."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError("Albumentations is required to build image transforms.") from exc

    normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    base = A.Compose([A.Resize(image_size, image_size), normalize, ToTensorV2()])
    minority = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=0.1,
                rotate_limit=180,
                border_mode=0,
                p=0.8,
            ),
            A.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.0, p=0.5),
            normalize,
            ToTensorV2(),
        ]
    )
    return base, minority


def _ensure_tensor_image(image: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image.float()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected image with 3 dimensions, got shape {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    label: int
    lesion_id: str | None


class DermoscopyDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        image_root: str | Path,
        *,
        image_column: str = "image_id",
        label_column: str = "dx",
        group_column: str = "lesion_id",
        transform: object | None = None,
        minority_transform: object | None = None,
        minority_classes: Iterable[str] = ("df", "vasc", "akiec", "bcc"),
        class_to_index: Mapping[str, int] | None = None,
    ) -> None:
        _require_columns(metadata, [image_column, label_column])
        self.metadata = metadata.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.image_column = image_column
        self.label_column = label_column
        self.group_column = group_column
        self.transform = transform
        self.minority_transform = minority_transform or transform
        self.minority_classes = set(minority_classes)
        names = class_to_index or {name: index for index, name in enumerate(DEFAULT_CLASS_NAMES)}
        self.class_to_index = dict(names)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        image_path = self.image_root / f"{row[self.image_column]}.jpg"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        transform = self.minority_transform if row[self.label_column] in self.minority_classes else self.transform
        if transform is not None:
            transformed = transform(image=image)
            image_tensor = _ensure_tensor_image(transformed["image"])
        else:
            image_tensor = _ensure_tensor_image(image)
        target = int(self.class_to_index[row[self.label_column]])
        return image_tensor, target
