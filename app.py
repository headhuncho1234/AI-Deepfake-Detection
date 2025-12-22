"""
Flask web application for Deepfake Detection.
Provides a REST API and web UI for uploading images and detecting deepfakes.
"""

import os
import sys
from pathlib import Path
import io
import base64
from datetime import datetime

from flask import Flask, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


# Utility: make objects JSON serializable (convert numpy types)
def make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    # numpy scalar
    try:
        import numpy as _np
    except Exception:
        _np = None

    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj

# Configure app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Load model globally
MODEL = None
MODEL_PATH = Path(__file__).parent / 'models' / 'trained_models' / 'baseline_best.keras'


def load_model():
    """Load the trained deepfake detection model."""
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please train the model first using: python models/train_cnn.py"
            )
        try:
            MODEL = keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    return MODEL


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Load and preprocess image for model inference."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to model input size
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numpy array (keep as [0, 255] range as per EfficientNetB0)
    img_array = np.array(img, dtype=np.float32)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img


def predict_image(image_path):
    """Predict whether image is real or fake."""
    model = load_model()
    
    # Preprocess
    img_array, pil_img = preprocess_image(image_path)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Convert to probability
    fake_probability = float(prediction)
    real_probability = 1.0 - fake_probability
    
    # Determine label
    is_fake = fake_probability > 0.5
    label = "FAKE" if is_fake else "REAL"
    confidence = float(max(fake_probability, real_probability))

    # Ensure all values are native Python types (JSON serializable)
    return {
        'label': label,
        'confidence': confidence,
        'fake_prob': float(fake_probability),
        'real_prob': float(real_probability),
        'raw_prediction': float(prediction)
    }


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for image prediction."""
    # Using a try/finally to ensure temp file removal and robust JSON encoding
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
    filepath = app.config['UPLOAD_FOLDER'] / filename
    file.save(filepath)

    try:
        result = predict_image(filepath)
        result = make_json_serializable(result)

        # Debug: log result types to help diagnose serialization issues
        try:
            debug_info = {k: type(v).__name__ for k, v in result.items()}
        except Exception:
            debug_info = repr(result)
        print('DEBUG /api/predict result types ->', debug_info)

        # Build response using stdlib json to avoid framework-specific encoders
        import json
        response_body = {
            'success': True,
            'result': result,
            'filename': file.filename
        }
        return current_app.response_class(json.dumps(response_body), mimetype='application/json')

    except Exception as e:
        # Log full traceback to server log for debugging
        import traceback
        tb = traceback.format_exc()
        print('Error during /api/predict:', tb)
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            if filepath.exists():
                filepath.unlink()
        except Exception:
            pass


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        model_exists = MODEL_PATH.exists()
        model_loaded = MODEL is not None
        return jsonify({
            'status': 'ok',
            'model_path': str(MODEL_PATH),
            'model_exists': model_exists,
            'model_loaded': model_loaded
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/info', methods=['GET'])
def info():
    """Get model info."""
    try:
        model = load_model()
        return jsonify({
            'model_name': 'EfficientNetB0 (Transfer Learning)',
            'input_shape': (224, 224, 3),
            'output': 'Binary classification (Real vs. Fake)',
            'framework': 'TensorFlow/Keras',
            'model_path': str(MODEL_PATH)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File too large (max 50MB)'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎬 Deepfake Detection Web Server")
    print("="*60)
    
    # Check if model exists
    if MODEL_PATH.exists():
        try:
            load_model()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load model: {e}")
            print("  The server will start but predictions will fail.")
    else:
        print(f"⚠ Model not found at: {MODEL_PATH}")
        print("  To train a model, run: python models/train_cnn.py")
        print("  The server will start, but predictions will fail without a model.\n")
    
    print(f"\n✓ Starting server at http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop\n")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=False, host='127.0.0.1', port=8000, use_reloader=False)
