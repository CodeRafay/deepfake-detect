---
title: deepfake-detect
emoji: 🔍
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
license: mit
models:
  - CodeRafay/deepfake-detect
short_description: Detect face deepfakes in images and videos
---

Upload an image or video to get a fake score per face, with a per-frame chart for videos.
It uses the latest model from [CodeRafay/deepfake-detect](https://huggingface.co/CodeRafay/deepfake-detect): unseen-dataset (Celeb-DF v2) AUC 0.914.

**A score is evidence, not proof.** The model is for research use only. Code and full write-up: https://github.com/CodeRafay/deepfake-detect
