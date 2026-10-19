"""Checks that need no model weights. Run: python -m pytest tests  (or python tests/test_core.py)"""
import tempfile
from pathlib import Path

import numpy as np

from deepfake_detect.faces import square_box
from deepfake_detect.model import load_head, save_head
from deepfake_detect.train import per_file


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


def test_per_file_averages_frames():
    p, y = per_file(np.array([0.2, 0.4, 0.9]), np.array([0, 0, 1]), np.array([0, 0, 1]))
    assert np.allclose(p, [0.3, 0.9]) and list(y) == [0, 1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
