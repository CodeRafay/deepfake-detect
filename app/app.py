"""
Flask Web Application for Deepfake Detection

Provides:
- Web interface for image upload
- REST API for predictions
- Grad-CAM heatmap visualization
- Configurable preprocessing and analysis parameters
"""

from src.module1_preproc import (
    denoise_nlm, denoise_bilateral, denoise_median, gamma_correction,
    clahe_enhancement, histogram_equalization, jpeg_compress,
    adjust_brightness_contrast, sharpen_image, add_blur
)
from src.gradcam import GradCAM, heatmap_to_base64
from src.infer import load_model, predict, preprocess_image
import tempfile
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
from pathlib import Path

# Add parent directory to path BEFORE other imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import Flask and other standard libraries

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Model (loaded lazily)
model = None
model_path = None
model_cache = {}  # Cache for multiple models


def get_available_models():
    """Get list of available models in the models directory."""
    models_dir = Path(__file__).parent.parent / 'models'
    if not models_dir.exists():
        return []

    model_files = []
    for ext in ['*.pth', '*.pkl', '*.h5']:
        model_files.extend(models_dir.glob(ext))

    return sorted([f.name for f in model_files])


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model(model_name=None):
    """Get or load the model."""
    global model, model_path, model_cache

    # If specific model requested
    if model_name:
        # Check cache first
        if model_name in model_cache:
            return model_cache[model_name]

        # Load the requested model
        requested_path = Path(__file__).parent.parent / 'models' / model_name
        if requested_path.exists():
            print(f"Loading model from {requested_path}")
            loaded_model = load_model(str(requested_path))
            model_cache[model_name] = loaded_model
            return loaded_model
        else:
            print(f"WARNING: Requested model {model_name} not found.")
            return None

    # Default: load any available model
    if model is not None:
        return model

    # Try to find model
    possible_paths = [
        Path(__file__).parent.parent / 'models' / 'best_model.pth',
        Path(__file__).parent.parent / 'models' / 'final_model.pth',
        Path('models/best_model.pth'),
        Path('models/final_model.pth'),
    ]

    for path in possible_paths:
        if path.exists():
            model_path = str(path)
            print(f"Loading model from {model_path}")
            model = load_model(model_path)
            return model

    # No model found - will return None
    print("WARNING: No trained model found. Please train a model first.")
    return None


@app.route('/')
def index():
    """Render main page."""
    available_models = get_available_models()
    return render_template('index.html', available_models=available_models)


@app.route('/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for deepfake detection with preprocessing options.

    Accepts: multipart/form-data with 'image' file and optional parameters:
        - denoise_strength (0-20): NLM denoising strength
        - contrast_mode (none/clahe/histeq): Contrast enhancement
        - gamma (0.5-2.0): Gamma correction
        - threshold (0-100): Classification threshold percentage
        - heatmap_opacity (0-100): Heatmap overlay opacity
        - simulate_compression (0-100): JPEG compression quality (0=off)

    Returns: JSON with prob, label, confidence, heatmap_b64
    """
    # Check if model is loaded
    model = get_model()
    if model is None:
        return jsonify({
            'error': 'No model loaded. Please train a model first.',
            'success': False
        }), 500

    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'success': False
        }), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'success': False
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
            'success': False
        }), 400

    try:
        # Get model selection
        selected_model = request.form.get('model', None)

        # Get preprocessing parameters from form
        denoise_method = request.form.get('denoise_method', 'none')
        denoise_strength = float(request.form.get('denoise_strength', 0))
        contrast_mode = request.form.get('contrast_mode', 'none')
        gamma_value = float(request.form.get('gamma', 1.0))
        brightness_value = float(request.form.get('brightness', 0))
        contrast_value = float(request.form.get('contrast', 1.0))
        sharpen_value = float(request.form.get('sharpen', 0))
        blur_value = int(request.form.get('blur', 0))
        threshold = float(request.form.get('threshold', 50)) / 100.0
        heatmap_opacity = float(request.form.get(
            'heatmap_opacity', 50)) / 100.0
        compression_quality = int(request.form.get('simulate_compression', 0))

        # Save temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        # Load and preprocess image
        img = cv2.imread(temp_path)
        original_img = img.copy()
        preprocessed = False

        # Apply preprocessing based on parameters
        if compression_quality > 0 and compression_quality < 100:
            img = jpeg_compress(img, quality=compression_quality)
            preprocessed = True

        if blur_value > 0:
            img = add_blur(img, kernel_size=blur_value)
            preprocessed = True

        if denoise_strength > 0 and denoise_method != 'none':
            if denoise_method == 'nlm':
                img = denoise_nlm(img, h=denoise_strength)
            elif denoise_method == 'bilateral':
                img = denoise_bilateral(
                    img, d=9, sigma_color=denoise_strength*7.5, sigma_space=denoise_strength*7.5)
            elif denoise_method == 'median':
                kernel = max(3, int(denoise_strength/2) * 2 + 1)  # odd number
                img = denoise_median(img, kernel_size=kernel)
            preprocessed = True

        if brightness_value != 0 or contrast_value != 1.0:
            img = adjust_brightness_contrast(
                img, brightness=brightness_value, contrast=contrast_value)
            preprocessed = True

        if gamma_value != 1.0:
            img = gamma_correction(img, gamma=gamma_value)
            preprocessed = True

        if sharpen_value > 0:
            img = sharpen_image(img, strength=sharpen_value)
            preprocessed = True

        if contrast_mode == 'clahe':
            img = clahe_enhancement(img)
            preprocessed = True
        elif contrast_mode == 'histeq':
            img = histogram_equalization(img)
            preprocessed = True

        # Save preprocessed image for prediction
        if preprocessed:
            preprocessed_path = os.path.join(
                app.config['UPLOAD_FOLDER'], 'preprocessed_' + filename)
            cv2.imwrite(preprocessed_path, img)
            predict_path = preprocessed_path
        else:
            predict_path = temp_path

        # Load selected model if specified
        if selected_model:
            current_model = get_model(selected_model)
            if current_model is None:
                return jsonify({
                    'error': f'Failed to load model: {selected_model}',
                    'success': False
                }), 500
        else:
            current_model = model

        # Make prediction
        result = predict(current_model, predict_path, generate_heatmap=True)

        # Apply custom threshold
        prob = result['prob']
        label = 'fake' if prob > threshold else 'real'
        confidence = (prob if prob > threshold else (1 - prob)) * 100

        # Regenerate heatmap with custom opacity if needed
        if heatmap_opacity != 0.5 and 'heatmap' in result:
            heatmap_b64 = heatmap_to_base64(
                result['heatmap'], img, alpha=heatmap_opacity)
        else:
            heatmap_b64 = result.get('heatmap_b64', '')

        # Convert preprocessed image to base64 if preprocessing was applied
        preprocessed_b64 = ''
        if preprocessed:
            _, buffer = cv2.imencode('.jpg', img)
            preprocessed_b64 = base64.b64encode(buffer).decode('utf-8')
            preprocessed_b64 = f'data:image/jpeg;base64,{preprocessed_b64}'
            print(
                f"Preprocessed image encoded, length: {len(preprocessed_b64)}")
        else:
            print("No preprocessing applied")

        # Clean up
        os.remove(temp_path)
        if preprocessed:
            os.remove(preprocessed_path)

        return jsonify({
            'success': True,
            'prob': prob,
            'label': label,
            'confidence': confidence,
            'heatmap_b64': heatmap_b64,
            'preprocessed_b64': preprocessed_b64,
            'threshold': threshold,
            'preprocessed': preprocessed,
            'model_used': selected_model if selected_model else 'default',
            'settings': {
                'model': selected_model if selected_model else 'default',
                'denoise_strength': denoise_strength,
                'contrast_mode': contrast_mode,
                'gamma': gamma_value,
                'threshold': threshold * 100,
                'heatmap_opacity': heatmap_opacity * 100,
                'compression_quality': compression_quality
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/predict_base64', methods=['POST'])
def api_predict_base64():
    """
    Alternative API endpoint accepting base64 encoded image.

    Accepts: JSON with 'image' base64 string
    Returns: JSON with prob, label, confidence, heatmap_b64
    """
    model = get_model()
    if model is None:
        return jsonify({
            'error': 'No model loaded',
            'success': False
        }), 500

    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({
            'error': 'No image data provided',
            'success': False
        }), 400

    try:
        # Decode base64
        img_data = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_data))

        # Save temporarily
        temp_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'temp_upload.png')
        img.save(temp_path)

        # Make prediction
        result = predict(model, temp_path, generate_heatmap=True)

        # Clean up
        os.remove(temp_path)

        return jsonify({
            'success': True,
            'prob': result['prob'],
            'label': result['label'],
            'confidence': result['confidence'],
            'heatmap_b64': result.get('heatmap_b64', '')
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    model = get_model()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': model_path if model is not None else None
    })


@app.route('/api/info')
def api_info():
    """API information endpoint."""
    return jsonify({
        'name': 'Deepfake Detection API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'Web interface',
            '/predict': 'POST - Upload image for prediction',
            '/predict_base64': 'POST - Send base64 image for prediction',
            '/health': 'GET - Health check',
            '/api/info': 'GET - API information',
            '/api/models': 'GET - List available models'
        },
        'allowed_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB'
    })


@app.route('/api/models')
def api_models():
    """Get list of available models."""
    models = get_available_models()
    return jsonify({
        'success': True,
        'models': models,
        'count': len(models)
    })


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large. Maximum size is 16MB.',
        'success': False
    }), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Deepfake Detection Web Application")
    print("=" * 60)

    # Try to load model on startup
    model = get_model()
    if model is None:
        print("\nWARNING: No model found!")
        print("Please train a model first using:")
        print("  python src/train.py")
        print("\nThe app will still run but predictions will fail.")

    print("\nStarting server at http://127.0.0.1:5000/")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
