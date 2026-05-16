import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from derma_hybrid_model.data import DEFAULT_CLASS_NAMES, create_train_val_test_split
from derma_hybrid_model.explainability import generate_integrated_gradients, summarize_attributions
from derma_hybrid_model.losses import MulticlassFocalLoss
from derma_hybrid_model.model import HybridDermClassifier
from derma_hybrid_model.train import cache_fused_features


class FakeBackbone(torch.nn.Module):
    def __init__(self, features: int, scale: float) -> None:
        super().__init__()
        self.num_features = features
        self.scale = scale

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        base = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return base.repeat(1, self.num_features) * self.scale


class FakeIntegratedGradients:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def attribute(self, image, baselines, target, n_steps, return_convergence_delta):
        del baselines, target, n_steps, return_convergence_delta
        return torch.ones_like(image), torch.tensor([0.0])


class PipelineTests(unittest.TestCase):
    def _metadata(self) -> pd.DataFrame:
        rows = []
        for class_index, class_name in enumerate(DEFAULT_CLASS_NAMES):
            for lesion_offset in range(5):
                lesion_id = f"{class_name}-lesion-{lesion_offset}"
                for image_offset in range(2):
                    rows.append(
                        {
                            "image_id": f"{class_name}-{lesion_offset}-{image_offset}",
                            "dx": class_name,
                            "lesion_id": lesion_id,
                            "meta_index": class_index,
                        }
                    )
        return pd.DataFrame(rows)

    def test_grouped_split_keeps_lesions_disjoint(self) -> None:
        metadata = self._metadata()
        splits = create_train_val_test_split(metadata, n_splits=5, val_fold=0, test_fold=1, seed=123)

        train_groups = set(splits["train"]["lesion_id"])
        val_groups = set(splits["val"]["lesion_id"])
        test_groups = set(splits["test"]["lesion_id"])

        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))
        self.assertEqual(len(metadata), sum(len(frame) for frame in splits.values()))

    def test_hybrid_model_fuses_backbone_vectors(self) -> None:
        backbones = {
            "mobile": FakeBackbone(3, 1.0),
            "efficient": FakeBackbone(5, 2.0),
            "resnet": FakeBackbone(7, 3.0),
            "densenet": FakeBackbone(11, 4.0),
        }
        model = HybridDermClassifier(num_classes=7, pretrained=False, backbones=backbones)
        images = torch.randn(2, 3, 8, 8)

        fused = model.extract_features(images)
        logits = model(images)

        self.assertEqual(fused.shape, (2, 26))
        self.assertEqual(logits.shape, (2, 7))

    def test_focal_loss_matches_cross_entropy_when_gamma_is_zero(self) -> None:
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.5, -0.3]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)

        focal = MulticlassFocalLoss(gamma=0.0)
        self.assertTrue(torch.allclose(focal(logits, targets), F.cross_entropy(logits, targets)))

    def test_integrated_gradients_helper_and_feature_cache(self) -> None:
        backbones = {"mobile": FakeBackbone(4, 1.0)}
        model = HybridDermClassifier(
            num_classes=2,
            pretrained=False,
            backbones=backbones,
            feature_dims={"mobile": 4},
        )
        image = torch.rand(1, 3, 6, 6)

        result = generate_integrated_gradients(
            model,
            image,
            integrated_gradients_cls=FakeIntegratedGradients,
            steps=8,
        )

        heatmap = summarize_attributions(result["attributions"])
        self.assertEqual(result["target"], int(model(image).argmax(dim=1).item()))
        self.assertEqual(heatmap.shape, (6, 6))
        self.assertAlmostEqual(float(heatmap.max()), 0.0)

        dataloader = [(image, torch.tensor([1], dtype=torch.long))]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = cache_fused_features(model, dataloader, Path(tmpdir) / "cache.pt", device="cpu")
            payload = torch.load(output)
            self.assertEqual(tuple(payload["features"].shape), (1, 4))
            self.assertEqual(payload["targets"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
