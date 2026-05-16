from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_TIMM_BACKBONES = OrderedDict(
    [
        ("mobile", "mobilenetv2_100"),
        ("efficient", "efficientnet_b2"),
        ("resnet", "resnet50"),
        ("densenet", "densenet121"),
    ]
)


def _create_default_backbones(pretrained: bool) -> tuple[nn.ModuleDict, dict[str, int]]:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required to create the default pretrained backbones.") from exc

    backbones = nn.ModuleDict()
    feature_dims: dict[str, int] = {}
    for key, model_name in DEFAULT_TIMM_BACKBONES.items():
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feature_dim = getattr(backbone, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"Backbone {model_name!r} does not expose num_features.")
        backbones[key] = backbone
        feature_dims[key] = int(feature_dim)
    return backbones, feature_dims


class HybridDermClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        *,
        pretrained: bool = True,
        backbones: Mapping[str, nn.Module] | None = None,
        feature_dims: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        if backbones is None:
            created_backbones, created_feature_dims = _create_default_backbones(pretrained)
            backbones = created_backbones
            feature_dims = created_feature_dims
        else:
            backbones = nn.ModuleDict(backbones)
            if feature_dims is None:
                feature_dims = {
                    name: int(getattr(module, "num_features"))
                    for name, module in backbones.items()
                }

        self.backbones = nn.ModuleDict(backbones)
        self.feature_dims = {name: int(size) for name, size in dict(feature_dims).items()}
        fused_dim = sum(self.feature_dims.values())
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def freeze_backbones(self) -> None:
        for parameter in self.backbones.parameters():
            parameter.requires_grad = False

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for backbone in self.backbones.values():
            features = backbone(images)
            if features.ndim > 2:
                features = torch.flatten(F.adaptive_avg_pool2d(features, output_size=1), 1)
            outputs.append(features)
        return torch.cat(outputs, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        fused_features = self.extract_features(images)
        return self.classifier(fused_features)
