# Deepfake Detection Pipeline

An end-to-end deepfake detection system featuring preprocessing, classical ML baseline, deep learning with Grad-CAM explainability, and a Flask web application.

## Features

- 🔬 **Module 1**: Image preprocessing, degradation simulation, and restoration
- 📊 **Module 2**: Classical ML baseline (HOG, LBP, color histograms + SVM/RF)
- 🧠 **Module 3**: ResNet18 fine-tuning with Grad-CAM explainability
- 🌐 **Web App**: Flask API with interactive parameter controls

## Quick Start

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/CodeRafay/deepfake-detect.git
cd deepfake-detect

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Option A**: Generate synthetic test data:

```bash
python3 src/generate_sample_data.py --train 100 --test 25
```

**Option B**: Use a real dataset (FaceForensics++, Celeb-DF, etc.):

```
data/
├── train/
│   ├── real/    # Authentic face images
│   └── fake/    # Manipulated images
└── test/
    ├── real/
    └── fake/
```

**Option C**: import the below dataset

import kagglehub

# Download latest version

```python
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")

print("Path to dataset files:", path)
```

```bash
# Train classical baseline (SVM)
python3 src/train_classical.py

# Train deep learning model
python3 src/train.py --epochs 20

# Quick test (2 epochs)
python3 src/train.py --epochs 2 --quick_test
```

### 4. Run Web App

```bash
python3 app/app.py
# Open http://127.0.0.1:5000/
```

## Web Interface Features

| Control              | Description                            |
| -------------------- | -------------------------------------- |
| Detection Threshold  | Adjust fake/real classification cutoff |
| Heatmap Opacity      | Control Grad-CAM overlay visibility    |
| Denoise Strength     | Apply NLM denoising before analysis    |
| Gamma Correction     | Adjust image brightness                |
| Contrast Enhancement | CLAHE or histogram equalization        |
| JPEG Compression     | Simulate compression artifacts         |

## Project Structure

```
deepfake-detect/
├── src/
│   ├── module1_preproc.py    # Preprocessing & restoration
│   ├── module2_features.py   # Feature extraction & classical ML
│   ├── module3_model.py      # Deep learning model
│   ├── gradcam.py            # Grad-CAM explainability
│   ├── train.py              # DL training script
│   ├── train_classical.py    # Classical baseline training
│   └── infer.py              # Inference utilities
├── app/
│   ├── app.py                # Flask API
│   └── templates/index.html  # Web interface
├── configs/config.yaml       # Training configuration
├── reports/                  # Generated figures & metrics
└── requirements.txt          # Dependencies
```

## API Endpoints

| Endpoint    | Method | Description                 |
| ----------- | ------ | --------------------------- |
| `/`         | GET    | Web interface               |
| `/predict`  | POST   | Upload image for prediction |
| `/health`   | GET    | Service status              |
| `/api/info` | GET    | API information             |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

## License

MIT License
