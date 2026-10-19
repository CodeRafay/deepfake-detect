"""v0.2 training: tune CLIP's LayerNorms + a linear head on face crops, with self-blended fakes.

Two changes over the frozen linear probe in train.py:
- LayerNorm tuning: only CLIP's ~100k LayerNorm parameters (0.03% of the model) and the head are trained.
  Enough to bend the features toward forensic cues, too few to memorise FF++'s four manipulation methods.
- Self-Blended Images (Shiohara & Yamasaki, CVPR 2022): extra fakes made on the fly by blending a real face
  with a slightly altered copy of itself. They carry the blend boundary and colour mismatch that nearly every
  face swap has, but no fingerprint of any particular generator.

Same data layout as train.py. Face crops are extracted once (in parallel, on CPU) into --crops and reused.

    python -m deepfake_detect.finetune --data data --eval celebdf=celebdf --out head.safetensors
    python -m deepfake_detect.finetune ... --fakes sbi     # ablation: real faces + SBI only, no FF++ fakes
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .faces import IMAGE_EXTS, VIDEO_EXTS, crop, detect_faces, load_image, read_frames
from .model import DEFAULT_BACKBONE, Backbone, save_head
from .train import evaluate, per_file, pick_threshold, print_table, sigmoid


# ---------- face crops: extracted once, lossless, at native size (exactly what Detector feeds CLIP) ----------

def _extract(job):
    path, out_dir, frames = job
    try:
        imgs = [f for _, f in read_frames(path, frames)] if path.suffix.lower() in VIDEO_EXTS else [load_image(path)]
    except ValueError:
        return 0
    n = 0
    for i, im in enumerate(imgs):
        boxes = detect_faces(im)
        if boxes:
            bgr = cv2.cvtColor(crop(im, boxes[0]), cv2.COLOR_RGB2BGR)
            cv2.imencode(".png", bgr)[1].tofile(str(out_dir / f"{path.stem}__{i}.png"))
            n += 1
    return n


def extract_crops(folder: Path, out: Path, frames: int):
    """Largest face of every image / sampled frame -> out/{real,fake}/<file>__<frame>.png. Skipped if done."""
    if (out / ".done").exists():
        return
    jobs = []
    for cls in ("real", "fake"):
        (out / cls).mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in (folder / cls).rglob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        if not files:
            raise SystemExit(f"no images or videos in {folder / cls}")
        jobs += [(p, out / cls, frames) for p in files]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        n = sum(tqdm(ex.map(_extract, jobs, chunksize=4), total=len(jobs), desc=f"crops {folder.name}"))
    (out / ".done").touch()
    print(f"{folder}: {n} face crops from {len(jobs)} files")


def list_crops(out: Path) -> list:
    """[(path, label, source file)] with fake=1; the source file groups frames for per-video metrics."""
    return [(p, label, f"{cls}/{p.stem.rsplit('__', 1)[0]}")
            for label, cls in enumerate(("real", "fake")) for p in sorted((out / cls).glob("*.png"))]


# ---------- augmentation ----------

def self_blend(img: np.ndarray, rng) -> np.ndarray:
    """Self-Blended Image: paste a colour-shifted, slightly moved and blurred copy of a real face onto itself
    through a soft, irregular face-shaped mask. Crops are the face box grown by 30% per side, so the face
    sits in the central ~60% of the crop."""
    h, w = img.shape[:2]
    src = img.astype(np.float32) * rng.uniform(0.9, 1.1, 3) + rng.uniform(-15, 15)
    if rng.random() < 0.5:
        src = cv2.GaussianBlur(src, (0, 0), rng.uniform(0.5, 1.5))
    s, (dx, dy) = rng.uniform(0.95, 1.05), rng.uniform(-0.03, 0.03, 2) * (w, h)
    src = cv2.warpAffine(src, np.float32([[s, 0, (1 - s) * w / 2 + dx], [0, s, (1 - s) * h / 2 + dy]]), (w, h),
                         borderMode=cv2.BORDER_REFLECT)

    mask = np.zeros((h, w), np.float32)
    center = (round(w * rng.uniform(0.47, 0.53)), round(h * rng.uniform(0.50, 0.56)))
    axes = (round(w * rng.uniform(0.22, 0.30)), round(h * rng.uniform(0.28, 0.36)))
    cv2.ellipse(mask, center, axes, rng.uniform(-10, 10), 0, 360, 1.0, -1)
    warp = cv2.resize(rng.uniform(-1, 1, (4, 4, 2)).astype(np.float32), (w, h)) * 0.04 * w  # wobble the edge
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mask = cv2.remap(mask, gx + warp[..., 0], gy + warp[..., 1], cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.02, 0.08) * w)
    mask = (mask * rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0]))[..., None]  # blend strength, as in the paper
    # rint, not truncation: the blurred mask is ~1e-9 far from the face, and truncating 99.999 -> 99 would
    # darken every SBI fake by one level - a shortcut the model would happily learn instead of the blend
    return np.rint(src * mask + img * (1 - mask)).clip(0, 255).astype(np.uint8)


def augment(img: np.ndarray, rng) -> np.ndarray:
    """Flip, rescale and JPEG re-encode. Applied to both classes alike so none becomes a shortcut,
    and so scores hold up on re-uploaded media."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.2:
        h, w = img.shape[:2]
        s = rng.uniform(0.5, 0.9)
        img = cv2.resize(cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), (w, h))
    if rng.random() < 0.3:
        buf = cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]),
                           [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])[1]
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(img)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, items, processor, sbi_prob=0.0, train=False):
        self.items, self.processor, self.sbi_prob, self.train = items, processor, sbi_prob, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        img = load_image(path)
        if self.train:
            rng = np.random.default_rng()  # fresh OS entropy per sample, so workers don't repeat each other
            if label == 0 and rng.random() < self.sbi_prob:
                img, label = self_blend(img, rng), 1
            img = augment(img, rng)
        return self.processor(images=img, return_tensors="pt").pixel_values[0], label


# ---------- model ----------

class Tuned(torch.nn.Module):
    """CLIP with only its LayerNorms trainable, plus the same linear head as v0.1."""

    def __init__(self, name):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.clip = CLIPVisionModelWithProjection.from_pretrained(name)  # fp32 master weights; autocast does fp16
        self.clip.requires_grad_(False)
        for m in self.clip.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.requires_grad_(True)
        # recompute activations in the backward pass: ViT-L at batch 32 doesn't fit a 16 GB T4 otherwise
        self.clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = torch.nn.Linear(self.clip.config.projection_dim, 1)

    def forward(self, x):
        e = torch.nn.functional.normalize(self.clip(pixel_values=x).image_embeds.float(), dim=-1)
        return self.head(e).squeeze(-1)

    def tuned_state(self) -> dict:
        return {k: v.detach().float().cpu().numpy().copy() for k, v in self.clip.named_parameters() if v.requires_grad}


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            out.append(torch.sigmoid(model(x.to(device))).float().cpu())
    return torch.cat(out).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with train/ val/ [test/]")
    ap.add_argument("--eval", action="append", default=[], metavar="NAME=DIR", help="extra test set (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("head.safetensors"))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--frames", type=int, default=8, help="frames sampled per video")
    ap.add_argument("--crops", type=Path, default=Path("crops"), help="face crop cache folder")
    ap.add_argument("--fakes", choices=["both", "sbi", "ffpp"], default="both",
                    help="training fakes: dataset fakes + SBI (default), SBI only, or dataset fakes only")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--epoch-size", type=int, default=20000, help="samples drawn per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="LayerNorm learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    a = ap.parse_args(argv)

    sets = {"train": a.data / "train", "val": a.data / "val"}
    if (a.data / "test").exists():
        sets["test"] = a.data / "test"
    sets.update(dict(e.split("=", 1) for e in a.eval))
    for name, folder in sets.items():  # before CUDA starts: worker processes fork
        extract_crops(Path(folder), a.crops / name, a.frames)
    items = {name: list_crops(a.crops / name) for name in sets}

    from transformers import CLIPImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPImageProcessor.from_pretrained(a.backbone)
    workers = os.cpu_count()

    # Draw shares: real / SBI-from-real / dataset fake = 1/3 each ("both"), 1/2 + 1/2 ("sbi"), 1/2 + 1/2 ("ffpp")
    train = [t for t in items["train"] if t[1] == 0] if a.fakes == "sbi" else items["train"]
    share_real = {"both": 2 / 3, "sbi": 1.0, "ffpp": 0.5}[a.fakes]
    n_real = sum(t[1] == 0 for t in train)
    n_fake = len(train) - n_real
    weights = [share_real / n_real if t[1] == 0 else (1 - share_real) / n_fake for t in train]
    loader = torch.utils.data.DataLoader(
        CropDataset(train, processor, sbi_prob=0.0 if a.fakes == "ffpp" else 0.5, train=True), batch_size=a.batch,
        sampler=torch.utils.data.WeightedRandomSampler(weights, a.epoch_size), num_workers=workers,
        pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(CropDataset(items["val"], processor), batch_size=64, num_workers=workers)
    yv = np.array([t[1] for t in items["val"]])
    gv = np.array([t[2] for t in items["val"]])

    model = Tuned(a.backbone).to(device)
    tuned = [p for p in model.clip.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in tuned):,} CLIP parameters + {sum(p.numel() for p in model.head.parameters())} head")
    opt = torch.optim.AdamW([{"params": tuned, "lr": a.lr}, {"params": model.head.parameters(), "lr": a.head_lr}],
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=a.epochs * len(loader),
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best = None
    for epoch in range(1, a.epochs + 1):
        model.train()
        bar = tqdm(loader, desc=f"epoch {epoch}/{a.epochs}")
        for x, y in bar:
            with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(x.to(device, non_blocking=True)), y.float().to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            bar.set_postfix(loss=f"{loss.item():.3f}")
        pf, yf = per_file(predict(model, val_loader, device), yv, gv)
        auc = roc_auc_score(yf, pf)
        print(f"epoch {epoch}: val AUC {auc:.4f}")
        if best is None or auc > best[0]:  # model selection on val only, never on test sets
            best = (auc, epoch, model.tuned_state(), model.head.weight.detach().float().cpu().numpy().reshape(-1),
                    float(model.head.bias.item()))

    _, best_epoch, state, w, b = best
    print(f"best epoch: {best_epoch}")
    del model, opt, loader
    torch.cuda.empty_cache()

    # Evaluate exactly what users will run: a fresh Backbone with the saved weights, as Detector builds it.
    backbone = Backbone(a.backbone, device, state)

    def embed(its):
        X = [backbone.embed([load_image(p) for p, _, _ in its[i:i + 256]]) for i in range(0, len(its), 256)]
        return np.concatenate(X), np.array([t[1] for t in its]), np.array([t[2] for t in its])

    emb = {name: embed(items[name]) for name in sets if name != "train"}
    Xv, yv, gv = emb["val"]
    threshold = pick_threshold(*per_file(sigmoid(Xv @ w + b), yv, gv))
    metrics = {name: evaluate(w, b, threshold, *e) for name, e in emb.items()}
    print_table(threshold, metrics)

    save_head(a.out, w, b, threshold, a.backbone, metrics, state)
    config = {k: str(v) for k, v in vars(a).items()} | {"best_epoch": best_epoch}
    Path(a.out).with_suffix(".metrics.json").write_text(json.dumps({**metrics, "config": config}, indent=2))
    print(f"\nsaved {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
