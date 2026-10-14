"""Reading media and finding faces (OpenCV YuNet, ~230 KB, downloaded once)."""
import functools
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
CACHE_DIR = Path.home() / ".cache" / "deepfake-detect"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def load_image(path) -> np.ndarray:
    """Read an image as RGB uint8. imdecode instead of imread so non-ASCII paths work on Windows."""
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_frames(path, n: int = 16) -> list:
    """Return up to `n` evenly spaced (seconds, RGB frame) pairs from a video."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    for i in np.unique(np.linspace(0, max(total - 1, 0), n).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            frames.append((i / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"cannot read video: {path}")
    return frames


@functools.lru_cache(maxsize=1)
def _detector():
    path = CACHE_DIR / "yunet_2023mar.onnx"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(YUNET_URL, tmp)
        tmp.replace(path)
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold=0.8)


def square_box(x, y, w, h, margin, img_w, img_h) -> tuple:
    """Square box around a face, grown by `margin` x face size on each side, clipped to the image.

    The margin matters: blending seams of face swaps sit at the face border, so a tight crop cuts them off.
    """
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (0.5 + margin)
    return (int(max(cx - half, 0)), int(max(cy - half, 0)),
            int(min(cx + half, img_w)), int(min(cy + half, img_h)))


def detect_faces(rgb: np.ndarray, margin: float = 0.3, min_size: int = 32, max_side: int = 1280) -> list:
    """Face boxes (x0, y0, x1, y1), largest first. Detection runs on a downscaled copy; crops come from full res."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(rgb, (round(w * scale), round(h * scale))) if scale < 1 else rgb
    det = _detector()
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    boxes = []
    for f in faces if faces is not None else []:
        x, y, fw, fh = f[:4] / scale
        if min(fw, fh) >= min_size:
            boxes.append(square_box(x, y, fw, fh, margin, w, h))
    return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def crop(rgb: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1]
