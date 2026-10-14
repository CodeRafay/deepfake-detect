# deepfake-detect

Detect face deepfakes in images and videos. The model is a frozen CLIP encoder plus a single linear layer, and its accuracy is reported on a dataset it never trained on.

The repo is small enough to read in an afternoon: about 400 lines across four files. It's meant both as a tool you can use and as a reference for building a detector without fooling yourself.

```
deepfake-detect suspicious.mp4 photo.jpg
FAKE  score=0.912  suspicious.mp4
REAL  score=0.087  photo.jpg
```

> **Status:** pretrained weights are not published yet. To train your own (free, ~1–2 h on Kaggle), see [Train](#train). The [Results](#results) section will be filled in from that run, whatever the numbers turn out to be.

## Install

```bash
pip install git+https://github.com/CodeRafay/deepfake-detect          # CLI + Python API
pip install "deepfake-detect[app] @ git+https://github.com/CodeRafay/deepfake-detect"  # + web demo
```

The first run downloads CLIP ViT-L/14 (~1.7 GB) and a 230 KB face detector. A GPU helps, but CPU works: about 1 s per face.

## Use

**Command line**

```bash
deepfake-detect video.mp4 img1.jpg img2.png       # one line per file
deepfake-detect video.mp4 --json                  # full result, incl. per-frame scores
deepfake-detect video.mp4 --head my_head.safetensors --frames 32
```

**Python**

```python
from deepfake_detect import Detector

det = Detector()                        # or Detector("my_head.safetensors")
r = det.predict("video.mp4")
r["label"], r["score"]                  # 'fake', 0.91
r["frames"]                             # [{'time': 0.0, 'score': 0.88, 'faces': 1}, ...]

det.predict("photo.jpg")["faces"]       # [{'box': [x0, y0, x1, y1], 'score': 0.12}, ...]
det.predict_image(rgb_numpy_array)      # already have pixels? skip the file
```

**Web demo**

```bash
streamlit run deepfake_detect/app.py
```

Upload an image or video. The demo shows the verdict, a box around each face colored by its score, and for videos, a chart of the score across the clip.

## How it works

```
image / video ──► sample frames ──► find faces ──► crop with margin ──► CLIP ViT-L/14 ──► linear ──► score per face
                  (16, evenly)       (YuNet)        (+30% each side)      (frozen)          layer
                                                                                                │
                                      image: max over faces  ◄──────────────────────────────────┤
                                      video: mean over frames of (max over faces)  ◄────────────┘
```

Each step is there for a reason:

1. **Look at faces, not whole images.** Face swaps change about 10% of the pixels. If you resize a 1080p frame to 224 px, the face becomes a blur and the artifacts disappear. Cropping first gives the model the face at full detail.
2. **Keep a margin around the face.** A face swap is blended into the original frame at the edge of the face. That seam is one of the strongest clues, and a tight crop cuts it off.
3. **Use frozen CLIP features, not a fine-tuned CNN.** A CNN fine-tuned on FaceForensics++ gets ~99% on FaceForensics++ and often close to chance on anything else. It learns the fingerprint of those four specific manipulation methods. CLIP was trained on 400M ordinary images and is never updated here, so the linear layer has to rely on general-purpose features. [Ojha et al., CVPR 2023](https://arxiv.org/abs/2302.10174) showed this generalizes much better to unseen generators. It also makes training cheap: embed every face once, then fit a logistic regression in seconds.
4. **A linear head.** The head has 769 parameters. With that few it can't memorize the training set, and you can open the `.safetensors` file and read the whole thing.
5. **Choose the threshold on validation data.** Both the regularization strength and the decision threshold are picked on the val split. The test sets are only ever used for the final report.
6. **Score each video, not each frame.** Frames from one video are strongly correlated, so a frame-level AUC overstates how good the model is. Both are reported, but the per-video number is the headline.

## Results

Train: FaceForensics++ c23 (Deepfakes, Face2Face, FaceSwap, NeuralTextures), official splits.

| test set | seen in training? | videos | AUC | AP | balanced acc | frame AUC |
|---|---|---|---|---|---|---|
| FF++ test | same dataset | — | — | — | — | — |
| Celeb-DF v2 | **no** | — | — | — | — | — |

*Pending the first training run.* Read the Celeb-DF row. The FF++ row only shows the model learned something. The Celeb-DF row shows whether it generalizes, and a detector you'd use on real content has to hold up on data it hasn't seen.

## Train

The notebook [notebooks/train_kaggle.ipynb](notebooks/train_kaggle.ipynb) runs everything on a free Kaggle GPU: official FF++ splits, training, Celeb-DF evaluation, and an optional upload to Hugging Face. Both datasets need an access request to their authors, which is free for research.

For your own data, put files in this layout (images, videos, or both):

```
data/train/{real,fake}/   data/val/{real,fake}/   data/test/{real,fake}/   (test is optional)
```

```bash
python -m deepfake_detect.train --data data --eval other=path/to/other_dataset --out head.safetensors
```

It prints a Markdown results table and saves the head, with its metrics stored inside the file. Embeddings are cached in `.embeddings/`, so you can re-run with different settings for free.

**Evaluate a trained head on your own data:**

```bash
python -m deepfake_detect.train --head head.safetensors --eval mine=path/to/folder_with_real_and_fake
```

## How detectors fool their own authors

These are the mistakes that make a deepfake detector look great on paper and fail on real content. The previous version of this repo made most of them, which is why it was rewritten.

| Mistake | What happens | How this repo avoids it |
|---|---|---|
| Splitting by frame instead of by person | The same face appears in train and test, so the model recognizes identities instead of fakes | Official splits by source video; the train script's docstring warns about it |
| Using the test set to choose the checkpoint or threshold | The test score is optimistic by construction | Val picks C and the threshold; test is only used to report |
| Testing only on the training dataset | Measures memorization of the generator's fingerprint, not detection | Always report a held-out dataset (Celeb-DF) |
| Resizing the whole frame to 224 px | The artifacts are destroyed before the model sees them | Detect the face at full resolution, then crop |
| Training on synthetic or toy data | 100% accuracy that means nothing | Only real benchmarks |
| Reporting frame-level accuracy for videos | Correlated frames inflate the number | Per-video metrics are the headline |
| Preprocessing that differs between training and inference | Silent accuracy drop in production | One `Backbone.embed` function used by both |

## Limitations

- **A score is evidence, not proof.** Don't accuse someone, or clear someone, based on this tool alone. Errors land on real people.
- **Faces only.** Fully generated scenes, or edits that don't involve a face, are out of scope.
- **New generators.** Generalization is better than with a fine-tuned CNN, but expect a drop on manipulation methods that look nothing like FF++, especially recent diffusion-based face swaps.
- **Compression.** Heavy re-encoding, like messaging apps or repeated uploads, erases artifacts. Results on c23 (light compression) don't transfer to c40 (heavy).
- **Partial fakes.** The video score is a mean over frames, so a 2-second fake inside a 60-second clip gets diluted. Check `frames` in the output for spikes.
- **Bias.** FF++ and Celeb-DF are not demographically balanced. Error rates may differ across groups, and this hasn't been measured yet. That would make a good contribution.

## Project layout

```
deepfake_detect/
  faces.py    read images/videos, detect & crop faces
  model.py    Backbone (frozen CLIP), head save/load, Detector (predict image/video)
  train.py    embed datasets, fit + evaluate the head
  cli.py      `deepfake-detect` command
  app.py      Streamlit demo
notebooks/train_kaggle.ipynb   end-to-end training on a free GPU
tests/test_core.py             checks that run without model weights
```

## Contributing

Contributions that make the numbers more honest are the most valuable: new held-out test sets (DFDC, DeeperForensics, recent diffusion face swaps), robustness checks against compression and resizing, and per-group error rates. Please include the results table that `train.py` prints with any PR that changes the model.

```bash
pip install -e ".[app]" pytest && pytest tests
```

## Citation

This project builds on:

- Ojha, Li, Lee. *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* CVPR 2023.
- Rössler et al. *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019.
- Li et al. *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics.* CVPR 2020.
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.

## License

MIT for the code. The CLIP weights have their own license (MIT, OpenAI). Datasets are under their authors' terms. Trained heads inherit the FF++ terms (research use).
