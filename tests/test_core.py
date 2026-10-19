"""Checks that need no model weights. Run: python -m pytest tests  (or python tests/test_core.py)"""
import tempfile
from pathlib import Path

import numpy as np

from deepfake_detect.faces import square_box
from deepfake_detect.finetune import augment, self_blend
from deepfake_detect.model import load_head, save_head
from deepfake_detect.train import per_file, pick_threshold


def test_square_box_grows_and_clips():
    assert square_box(40, 40, 20, 20, 0.5, 100, 100) == (30, 30, 70, 70)  # 20px face -> 40px square
    assert square_box(0, 0, 20, 10, 0.3, 100, 100) == (0, 0, 26, 21)     # clipped at top-left, square on max side


def test_head_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "head.safetensors"
        save_head(path, np.arange(4.0), -0.5, 0.42, "some/backbone", {"val": {"auc": 0.9}})
        h = load_head(path)
    assert np.allclose(h["weight"], [0, 1, 2, 3]) and h["bias"] == -0.5
    assert h["threshold"] == 0.42 and h["backbone"] == "some/backbone" and h["metrics"]["val"]["auc"] == 0.9
    assert h["state"] == {}  # v0.1-style head: frozen backbone


def test_head_roundtrip_with_tuned_backbone():
    key = "vision_model.post_layernorm.weight"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "head.safetensors"
        save_head(path, np.zeros(4), 0.0, 0.5, "some/backbone", state={key: np.arange(3.0)})
        h = load_head(path)
    assert list(h["state"]) == [key] and np.allclose(h["state"][key], [0, 1, 2])


def test_per_file_averages_frames():
    p, y = per_file(np.array([0.2, 0.4, 0.9]), np.array([0, 0, 1]), np.array([0, 0, 1]))
    assert np.allclose(p, [0.3, 0.9]) and list(y) == [0, 1]


def test_pick_threshold_separates_classes():
    t = pick_threshold(np.array([0.1, 0.2, 0.7, 0.9]), np.array([0, 0, 1, 1]))
    assert 0.2 < t <= 0.7


def test_self_blend_changes_face_not_background():
    img = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    out = self_blend(img, np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[:8, :8], img[:8, :8])  # corners untouched (no truncation drift)
    assert np.abs(out[48:80, 48:80].astype(int) - img[48:80, 48:80]).mean() > 1  # face region altered


def test_augment_keeps_shape():
    img = np.random.default_rng(0).integers(0, 256, (90, 70, 3), dtype=np.uint8)
    for seed in range(20):
        out = augment(img, np.random.default_rng(seed))
        assert out.shape == img.shape and out.dtype == np.uint8 and out.flags.c_contiguous


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
