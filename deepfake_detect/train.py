"""Train the linear head on frozen CLIP face embeddings, then evaluate it honestly.

Data layout - every folder holds `real/` and `fake/`, with any mix of images and videos inside:

    DATA/train/{real,fake}   fit the head
    DATA/val/{real,fake}     choose C and the decision threshold  (never the test set)
    DATA/test/{real,fake}    optional, same-dataset test
    --eval NAME=DIR          any number of extra test sets, ideally from datasets you did NOT train on

Split by identity/source video, not by frame: frames of one person in both train and test is leakage,
and it is the most common reason deepfake detectors look great on paper and fail in practice.

    python -m deepfake_detect.train --data data --eval celebdf=celebdf_test --out head.safetensors
    python -m deepfake_detect.train --head head.safetensors --eval mydata=path/to/mydata   # evaluate only
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm

from .faces import IMAGE_EXTS, VIDEO_EXTS, crop, detect_faces, load_image, read_frames
from .model import DEFAULT_BACKBONE, Backbone, load_head, save_head


def embed_folder(backbone, folder: Path, frames: int, cache: Path):
    """Embed the largest face of every image / sampled video frame.
    Returns X (N, D), y (N,) with fake=1, g (N,) = index of the source file (for per-file metrics)."""
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"], d["g"]
    X, y, g, pending = [], [], [], []

    def flush():
        if pending:
            X.append(backbone.embed(pending))
            pending.clear()

    file_id = 0
    for label, cls in enumerate(["real", "fake"]):
        files = sorted(p for p in (folder / cls).rglob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        if not files:
            raise SystemExit(f"no images or videos in {folder / cls}")
        # ponytail: decode + face detection is single-threaded; a DataLoader with workers if this is the bottleneck
        for p in tqdm(files, desc=f"{folder.name}/{cls}"):
            try:
                imgs = [f for _, f in read_frames(p, frames)] if p.suffix.lower() in VIDEO_EXTS else [load_image(p)]
            except ValueError as e:
                print(f"skip: {e}")
                continue
            crops = [crop(im, boxes[0]) for im in imgs if (boxes := detect_faces(im))]
            pending += crops
            y += [label] * len(crops)
            g += [file_id] * len(crops)
            file_id += 1
            if len(pending) >= 256:
                flush()
    flush()
    if not y:
        raise SystemExit(f"no faces found anywhere in {folder}")
    X, y, g = np.concatenate(X), np.array(y), np.array(g)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, g=g)
    print(f"{folder}: {len(y)} faces from {len(np.unique(g))} files ({int(y.sum())} fake)")
    return X, y, g


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def per_file(p, y, g):
    """Average face scores per source file - the number that matters for videos."""
    ids = np.unique(g)
    return np.array([p[g == i].mean() for i in ids]), np.array([y[g == i][0] for i in ids])


def evaluate(w, b, threshold, X, y, g) -> dict:
    p = sigmoid(X @ w + b)
    pf, yf = per_file(p, y, g)
    return {"files": len(yf), "auc": roc_auc_score(yf, pf), "ap": average_precision_score(yf, pf),
            "balanced_acc": balanced_accuracy_score(yf, pf >= threshold), "frame_auc": roc_auc_score(y, p)}


def pick_threshold(pf, yf) -> float:
    """Youden's J on per-file validation scores: the cut that best separates real from fake."""
    fpr, tpr, thr = roc_curve(yf, pf)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))


def print_table(threshold, metrics):
    print(f"\nthreshold = {threshold:.4f} (chosen on val)\n")
    print("| set | files | AUC | AP | balanced acc | frame AUC |\n|---|---|---|---|---|---|")
    for name, m in metrics.items():
        print(f"| {name} | {m['files']} | {m['auc']:.3f} | {m['ap']:.3f} | {m['balanced_acc']:.3f} | {m['frame_auc']:.3f} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="folder with train/ val/ [test/]")
    ap.add_argument("--head", help="evaluate this existing head instead of training")
    ap.add_argument("--eval", action="append", default=[], metavar="NAME=DIR", help="extra test set (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("head.safetensors"))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--frames", type=int, default=8, help="frames sampled per video")
    ap.add_argument("--cache", type=Path, default=Path(".embeddings"), help="embedding cache folder")
    a = ap.parse_args(argv)
    if not a.data and not a.head:
        ap.error("pass --data to train or --head to evaluate")

    head = load_head(a.head) if a.head else None
    backbone = Backbone(head["backbone"], state=head["state"]) if head else Backbone(a.backbone)
    tag = backbone.name.replace("/", "_")
    if head and head["state"]:  # tuned backbone -> different embeddings, so a separate cache
        tag += "_" + hashlib.sha1(Path(a.head).read_bytes()).hexdigest()[:8]

    def load(name, folder):
        return embed_folder(backbone, Path(folder), a.frames, a.cache / f"{tag}_f{a.frames}_{name}.npz")

    if head:
        w, b, threshold = head["weight"], head["bias"], head["threshold"]
    else:
        Xtr, ytr, _ = load("train", a.data / "train")
        Xva, yva, gva = load("val", a.data / "val")
        best = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
            pf, yf = per_file(clf.predict_proba(Xva)[:, 1], yva, gva)
            auc = roc_auc_score(yf, pf)
            print(f"C={C:<5} val AUC={auc:.4f}")
            if best is None or auc > best[0]:
                best = (auc, clf, pf, yf)
        _, clf, pf, yf = best
        threshold = pick_threshold(pf, yf)
        w, b = clf.coef_[0], float(clf.intercept_[0])

    sets = ([("val", a.data / "val")] + ([("test", a.data / "test")] if (a.data / "test").exists() else [])
            if a.data else [])
    sets += [tuple(e.split("=", 1)) for e in a.eval]
    metrics = {name: evaluate(w, b, threshold, *load(name, folder)) for name, folder in sets}
    print_table(threshold, metrics)

    if not head:
        save_head(a.out, w, b, threshold, backbone.name, metrics)
        Path(a.out).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
