---
license: other
license_name: research-only
license_link: https://github.com/ondyari/FaceForensics
tags: [deepfake-detection, face-forensics, clip]
---

# deepfake-detect models

Trained heads for [deepfake-detect](https://github.com/CodeRafay/deepfake-detect): small classifiers that run on top of frozen `openai/clip-vit-large-patch14` face embeddings. Each version lives in its own folder and is never overwritten.

| version | what changed | FF++ test AUC | Celeb-DF v2 AUC (unseen) | size |
|---|---|---|---|---|
| [v0.1-clip-linear-probe](v0.1-clip-linear-probe) | baseline: frozen CLIP + logistic regression, 8 frames/video, FF++ c23 | 0.905 | 0.738 | 3.7 KB |

Every folder contains `head.safetensors`, the model, with its backbone, threshold and metrics stored inside the file, `head.metrics.json` with all numbers from the training run, and `training_run.ipynb`, the executed training notebook with its full log, so every result can be checked.

```bash
pip install git+https://github.com/CodeRafay/deepfake-detect
deepfake-detect video.mp4                                  # latest version
deepfake-detect video.mp4 --head v0.1-clip-linear-probe    # a specific version
```

**License:** research and non-commercial use only. The heads are trained on FaceForensics++, whose terms apply.
**Use responsibly:** a score is evidence, not proof. Don't make decisions about people based on it alone.
