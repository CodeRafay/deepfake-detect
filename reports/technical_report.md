# Deepfake Detection Pipeline - Technical Report

## 1. Executive Summary

This report documents an end-to-end deepfake detection pipeline implementing:
- **Module 1**: Image preprocessing and restoration with quality metrics
- **Module 2**: Classical ML baseline using HOG, LBP, and color features
- **Module 3**: Deep learning with ResNet18 and Grad-CAM explainability
- **Web Application**: Flask API with visual explanations

## 2. Dataset Selection & Characterization

### 2.1 Dataset Choice
The pipeline supports standard deepfake datasets:
- FaceForensics++ (FF++)
- Celeb-DF
- DFDC (subset)

**Rationale**: FF++ provides diverse manipulation methods (DeepFakes, Face2Face, FaceSwap) with multiple compression levels (c23, c40), enabling robustness testing.

### 2.2 Data Structure
```
data/
├── train/
│   ├── real/    # Authentic face images
│   └── fake/    # Manipulated images
└── test/
    ├── real/
    └── fake/
```

## 3. Module 1: Preprocessing & Restoration

### 3.1 Implemented Transforms

| Category | Functions |
|----------|-----------|
| **Degradation** | Gaussian noise, Salt & pepper, JPEG compression, Blur |
| **Geometric** | Rotation, Scaling, Flipping, Center crop |
| **Intensity** | Gamma correction, Histogram equalization, CLAHE |
| **Restoration** | NLM denoising, Bilateral filter, Median filter |

### 3.2 Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index (0-1)
- **MSE**: Mean Squared Error

### 3.3 Key Findings

| Degradation | PSNR (dB) | SSIM |
|-------------|-----------|------|
| JPEG Q=50 | ~32 | ~0.92 |
| JPEG Q=20 | ~28 | ~0.85 |
| Gaussian σ=25 | ~25 | ~0.78 |

**Insight**: Heavy compression (Q<20) significantly affects high-frequency forensic cues. NLM denoising provides 2-4dB PSNR improvement on Gaussian noise.

## 4. Module 2: Classical Features Baseline

### 4.1 Feature Extractors

| Feature | Dimension | Purpose |
|---------|-----------|---------|
| HOG | ~8100 | Edge/gradient structure |
| LBP | 26 | Local texture patterns |
| Color Histogram | 96 | Color distribution |
| Hu Moments | 7 | Shape invariants |
| Edge Statistics | 8 | Edge density/patterns |

### 4.2 Classifier Configuration
- **SVM**: RBF kernel, C=1.0
- **RandomForest**: 100 trees
- **Preprocessing**: StandardScaler → PCA (100 components)

### 4.3 Expected Baseline Performance
Based on similar studies, classical features achieve:
- Accuracy: 70-80%
- AUC: 0.75-0.85

*Actual results depend on dataset and manipulation type.*

## 5. Module 3: Deep Learning Model

### 5.1 Architecture
```
ResNet18 (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Linear (512 → 256) + ReLU
    ↓
Dropout (0.25)
    ↓
Linear (256 → 1) + Sigmoid
```

### 5.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Scheduler | Cosine Annealing |
| Early Stopping | 5 epochs patience |
| Batch Size | 32 |

### 5.3 Data Augmentation
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random crop (224×224 from 256×256)

### 5.4 Explainability: Grad-CAM
Gradient-weighted Class Activation Mapping visualizes:
- Which facial regions influence predictions
- Expected attention on eyes, mouth, blending boundaries
- Verification that model detects artifacts, not identity

## 6. Flask Web Application

### 6.1 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Upload image → prediction |
| `/health` | GET | Service status |

### 6.2 Response Format
```json
{
  "success": true,
  "prob": 0.87,
  "label": "fake",
  "confidence": 87.3,
  "heatmap_b64": "base64_encoded_image..."
}
```

## 7. Evaluation Metrics

### 7.1 Metrics Tracked
- **Accuracy**: Overall correctness
- **Precision**: Fake detection precision
- **Recall**: Fake detection rate
- **F1 Score**: Harmonic mean
- **AUC-ROC**: Area under ROC curve

### 7.2 Comparison Framework

| Method | Accuracy | AUC | F1 |
|--------|----------|-----|-----|
| Classical (SVM) | - | - | - |
| Classical (RF) | - | - | - |
| ResNet18 | - | - | - |

*Fill in after running experiments.*

## 8. Ablation Studies

### 8.1 Planned Experiments
1. **With vs. without restoration**: Does preprocessing help?
2. **Feature ablation**: Which classical features matter most?
3. **Compression robustness**: Performance on c23 vs c40

## 9. Limitations & Ethical Considerations

### 9.1 Technical Limitations
- Trained on specific manipulation methods
- May not generalize to unseen generation techniques
- Performance depends on image quality/compression

### 9.2 Ethical Notes
- Detection tools can be misused for censorship
- False positives may harm individuals
- Deepfake generation tools should be used responsibly
- Dataset bias may affect fairness

## 10. Reproduction Instructions

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate sample data (for testing)
python src/generate_sample_data.py

# Train classical baseline
python src/train_classical.py

# Train deep learning model
python src/train.py --epochs 20

# Run web app
python app/app.py
```

## 11. References

1. Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
2. Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR 2020
3. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
