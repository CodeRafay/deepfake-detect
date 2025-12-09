# Deepfake Detection Pipeline

An end-to-end deepfake detection system featuring preprocessing, classical ML baseline, deep learning with Grad-CAM explainability, and a Flask web application. The project supports training on custom datasets with flexible sample size control for faster experimentation.

## Features

- 🔬 **Module 1**: Image preprocessing, degradation simulation, and restoration
- 📊 **Module 2**: Classical ML baseline (HOG, LBP, color histograms + SVM/RF)
- 🧠 **Module 3**: ResNet18 fine-tuning with Grad-CAM explainability
- 🌐 **Web App**: Flask API with interactive parameter controls and Grad-CAM visualization
- 📥 **Dataset Import**: Automated Kaggle dataset download and organization
- ⚡ **Flexible Training**: Control training samples per class for quick experimentation

## Quick Start

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/CodeRafay/deepfake-detect.git
cd deepfake-detect

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

**Note**: If you need kagglehub for dataset import:

```bash
pip install kagglehub
```

### 2. Prepare Dataset

**Option A**: Import Kaggle Dataset (Recommended)

The project includes a script to automatically download and organize the Kaggle deepfake dataset:

```bash
python src/import_dataset.py
```

This will:

- Download the dataset from Kaggle: `manjilkarki/deepfake-and-real-images`
- Organize into `data/Train/Real`, `data/Train/Fake`, `data/Test/Real`, `data/Test/Fake`
- Merge validation set into test set
- Show statistics for all splits

**Option B**: Generate synthetic test data (for quick testing):

```bash
python src/generate_sample_data.py --train 100 --test 25
```

**Option C**: Use your own dataset with this structure:

```
data/
├── Train/
│   ├── Real/    # Authentic face images
│   └── Fake/    # Manipulated images
└── Test/
    ├── Real/
    └── Fake/
```

**Note**: The project supports both capitalized (`Real/Fake`) and lowercase (`real/fake`) folder names. You may use the below dataset too:

```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")

print("Path to dataset files:", path)
```

### 3. Train Models

**Classical ML Baseline (SVM/Random Forest):**

```bash
# Train with limited samples (faster, recommended for testing)
python src/train_classical.py --max_samples 5000

# Train with more samples for better accuracy
python src/train_classical.py --max_samples 10000 --classifier rf

# Use Random Forest with reduced PCA components (prevents overfitting)
python src/train_classical.py --max_samples 10000 --classifier rf --pca_components 100

# Train on ALL data (70,000+ samples per class, slower)
python src/train_classical.py
```

**Deep Learning Model (ResNet18):**

```bash
# Train with limited samples (faster, ~20-40 mins on CPU)
python src/train.py --max_samples 5000 --epochs 10

# Train with more data for better accuracy
python src/train.py --max_samples 10000 --epochs 15

# Quick test (minimal data, 2 epochs)
python src/train.py --quick_test

# Full training on all data
python src/train.py --epochs 20
```

**Training Parameters:**

- `--max_samples`: Limit samples per class (e.g., 5000 uses 5000 real + 5000 fake)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--classifier`: `svm` or `rf` (for classical training only)
- `--quick_test`: Fast test mode (100 samples, 2 epochs)

**Note**: Training on CPU will be significantly slower than GPU. With 5000 samples:

- Classical ML: ~10-20 minutes
- Deep Learning: ~30-60 minutes per epoch on CPU

### 4. Run Web App

Once you have a trained model, start the Flask web application:

```bash
python app/app.py
```

Then open your browser and navigate to: **http://127.0.0.1:5000/**

The app will automatically load the trained model from the `models/` directory. If no model is found, you can still access the web interface, but predictions will fail until you train a model.

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
│   ├── module1_preproc.py       # Preprocessing & restoration functions
│   ├── module2_features.py      # Feature extraction & classical ML
│   ├── module3_model.py         # Deep learning model (ResNet18)
│   ├── gradcam.py               # Grad-CAM explainability
│   ├── train.py                 # Deep learning training script
│   ├── train_classical.py       # Classical baseline training
│   ├── infer.py                 # Inference utilities
│   ├── utils.py                 # Helper functions
│   ├── generate_sample_data.py  # Generate synthetic test data
│   └── import_dataset.py        # Download & organize Kaggle dataset
├── app/
│   ├── app.py                   # Flask API server
│   └── templates/
│       └── index.html           # Web interface
├── configs/
│   └── config.yaml              # Training configuration
├── notebooks/
│   └── preprocessing_experiments.py  # Preprocessing analysis
├── data/                        # Dataset (not included in repo)
│   ├── Train/
│   │   ├── Real/
│   │   └── Fake/
│   └── Test/
│       ├── Real/
│       └── Fake/
├── models/                      # Trained models (not included)
│   ├── best_model.pth           # Deep learning model
│   └── classical_baseline.pkl   # Classical ML model
├── reports/                     # Training reports & metrics
│   ├── figures/                 # Plots & visualizations
│   └── metrics/                 # Performance metrics
├── .gitignore
├── README.md
└── requirements.txt             # Python dependencies
```

## API Endpoints

| Endpoint          | Method | Description                                    |
| ----------------- | ------ | ---------------------------------------------- |
| `/`               | GET    | Web interface (upload and analyze images)      |
| `/predict`        | POST   | Upload image for prediction with preprocessing |
| `/predict_base64` | POST   | Prediction with base64-encoded image           |
| `/health`         | GET    | Service health check                           |
| `/api/info`       | GET    | API version and configuration information      |

**Prediction Request Parameters:**

- `image`: Image file (multipart/form-data)
- `denoise_strength`: NLM denoising (0-20)
- `contrast_mode`: none/clahe/histeq
- `gamma`: Gamma correction (0.5-2.0)
- `threshold`: Classification threshold (0-100)
- `heatmap_opacity`: Grad-CAM overlay opacity (0-100)
- `simulate_compression`: JPEG compression quality (0-100)

## Performance Tips

### For Faster Training:

1. **Use smaller sample sizes**: Start with `--max_samples 1000` for quick tests
2. **Reduce epochs**: Use `--epochs 5` for initial experiments
3. **Use classical ML first**: SVM/RF trains faster than deep learning
4. **Enable GPU**: If available, PyTorch will automatically use CUDA

### For Better Accuracy:

1. **Use more training data**: `--max_samples 10000` or more
2. **Try Random Forest**: Often more robust than SVM
3. **Reduce PCA components**: `--pca_components 50` prevents overfitting
4. **Train longer**: Use `--epochs 20` for deep learning
5. **Monitor metrics**: Check `reports/metrics/` for training statistics

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'flask'`

- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Duplicate file counts (showing 2x images)

- **Solution**: Already fixed - code now uses sets to avoid duplicates on Windows

**Issue**: Model overfitting (high train accuracy, low test accuracy)

- **Solution**: Use more training data, reduce PCA components, or try Random Forest

**Issue**: Training is too slow

- **Solution**: Reduce `--max_samples` or use `--quick_test` mode

**Issue**: `pin_memory` warning when training

- **Solution**: Safe to ignore - it's just indicating no GPU is available

## Dataset Information

The project expects datasets with Real and Fake subfolders. Supported sources:

- **Kaggle**: `manjilkarki/deepfake-and-real-images` (automated import available)
- **FaceForensics++**: Popular benchmark dataset
- **Celeb-DF**: High-quality celebrity deepfakes
- **Custom datasets**: Any organized real/fake image collection

## Model Outputs

Training generates the following outputs:

**Classical ML:**

- `models/classical_baseline.pkl` - Trained SVM/RF model
- `reports/metrics/classical_baseline_*.txt` - Performance metrics
- `reports/figures/classical_confusion_matrix_*.png` - Confusion matrix
- `reports/figures/classical_roc_curve_*.png` - ROC curve

**Deep Learning:**

- `models/best_model.pth` - Best performing model checkpoint
- `models/final_model.pth` - Final epoch model
- `reports/metrics/training_*.csv` - Training history
- `reports/figures/training_curves_*.png` - Loss and accuracy plots

## Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3+
- scikit-image 0.21+
- Flask 2.3+
- OpenCV 4.8+
- See `requirements.txt` for complete list

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
