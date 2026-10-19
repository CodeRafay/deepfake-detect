"""The whole model: a frozen CLIP image encoder + one linear layer (logistic regression) on top.

CLIP is never fine-tuned. Its features were learned on 400M generic images, so they don't latch onto
the quirks of one fake-generation method the way a fine-tuned CNN does - that is what makes the
detector hold up on manipulation methods it never saw (Ojha et al., CVPR 2023, "UnivFD").
"""
import json
from pathlib import Path

import numpy as np

from .faces import VIDEO_EXTS, crop, detect_faces, load_image, read_frames

DEFAULT_BACKBONE = "openai/clip-vit-large-patch14"
MODELS_REPO = "CodeRafay/deepfake-detect"  # Hugging Face mirror of this repo's models/ folder
LATEST = "v0.1-clip-linear-probe"           # models/<version>/head.safetensors
LOCAL_MODELS = Path(__file__).resolve().parent.parent / "models"


class Backbone:
    """Frozen CLIP vision encoder. Training and inference both go through `embed`, so they can't drift apart."""

    def __init__(self, name: str = DEFAULT_BACKBONE, device: str = None):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self._torch = torch
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = CLIPImageProcessor.from_pretrained(name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()

    def embed(self, images: list, batch_size: int = 32) -> np.ndarray:
        """RGB uint8 arrays -> L2-normalised CLIP embeddings, shape (N, D)."""
        torch, out = self._torch, []
        for i in range(0, len(images), batch_size):
            px = self.processor(images=images[i:i + batch_size], return_tensors="pt").pixel_values
            with torch.no_grad():
                e = self.model(pixel_values=px.to(self.device, self.model.dtype)).image_embeds
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.projection_dim), np.float32)


def save_head(path, weight, bias, threshold, backbone, metrics=None):
    from safetensors.numpy import save_file

    save_file({"weight": np.asarray(weight, np.float32).reshape(-1), "bias": np.array([bias], np.float32)},
              str(path), metadata={"backbone": backbone, "threshold": str(threshold),
                                   "metrics": json.dumps(metrics or {})})


def load_head(path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), "np") as f:
        meta = f.metadata()
        return {"weight": f.get_tensor("weight"), "bias": float(f.get_tensor("bias")[0]),
                "threshold": float(meta["threshold"]), "backbone": meta["backbone"],
                "metrics": json.loads(meta.get("metrics", "{}"))}


def _resolve_head(head):
    """A .safetensors path, or a version name like 'v0.1-clip-linear-probe' (default: LATEST).
    Versions load from models/ in a cloned repo, otherwise from the Hugging Face mirror."""
    if head and Path(head).is_file():
        return head
    version = head or LATEST
    local = LOCAL_MODELS / version / "head.safetensors"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(MODELS_REPO, f"{version}/head.safetensors")
    except Exception as e:
        raise FileNotFoundError(f"'{version}' is neither a file nor a version in {MODELS_REPO}. "
                                "Train one with `python -m deepfake_detect.train` and pass its path.") from e


class Detector:
    """Scores images and videos. `score` is the head's fake probability; `label` compares it to the
    threshold chosen on the validation set during training."""

    def __init__(self, head: str = None, device: str = None):
        self.head = load_head(_resolve_head(head))
        self.threshold = self.head["threshold"]
        self.backbone = Backbone(self.head["backbone"], device)

    def score(self, crops: list) -> np.ndarray:
        z = self.backbone.embed(crops) @ self.head["weight"] + self.head["bias"]
        return 1 / (1 + np.exp(-z))

    def _result(self, score, **extra) -> dict:
        return {"label": "fake" if score >= self.threshold else "real", "score": round(float(score), 4),
                "threshold": round(self.threshold, 4), **extra}

    @staticmethod
    def _crops(rgb):
        boxes = detect_faces(rgb)
        return boxes, [crop(rgb, b) for b in boxes] or [rgb]  # no face: score the whole frame

    def predict_image(self, rgb: np.ndarray) -> dict:
        """One image. Any fake face makes the image fake, so the image score is the max over faces."""
        boxes, crops = self._crops(rgb)
        scores = self.score(crops)
        faces = [{"box": list(b), "score": round(float(s), 4)} for b, s in zip(boxes, scores)]
        extra = {} if boxes else {"note": "no face found; scored the whole image (less reliable)"}
        return self._result(scores.max(), faces=faces, **extra)

    def predict_video(self, path, frames: int = 16) -> dict:
        """Frame score = max over faces in the frame; video score = mean over sampled frames."""
        # ponytail: mean dilutes videos where only a few seconds are fake; per-frame scores are returned so you can see it
        items = [(t, *self._crops(rgb)) for t, rgb in read_frames(path, frames)]
        scores = self.score([c for _, _, crops in items for c in crops])
        per_frame, i = [], 0
        for t, boxes, crops in items:
            per_frame.append({"time": round(float(t), 2), "score": round(float(scores[i:i + len(crops)].max()), 4),
                              "faces": len(boxes)})
            i += len(crops)
        extra = {} if any(f["faces"] for f in per_frame) else {"note": "no face found in sampled frames"}
        return self._result(np.mean([f["score"] for f in per_frame]), frames=per_frame, **extra)

    def predict(self, path, frames: int = 16) -> dict:
        if Path(path).suffix.lower() in VIDEO_EXTS:
            return self.predict_video(path, frames)
        return self.predict_image(load_image(path))
