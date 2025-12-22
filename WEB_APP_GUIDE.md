# Web Interface - Deepfake Detection

## Overview
A Flask-based web application providing a user-friendly interface for deepfake detection. Upload images and get instant predictions with confidence scores.

## Features
- 📸 Drag-and-drop image upload
- 🔍 Real-time deepfake detection
- 📊 Confidence scores and probability breakdown
- 🎨 Modern, responsive UI
- 📱 Mobile-friendly design
- ⚡ Fast inference using pre-trained EfficientNetB0

## Getting Started

### 1. Install Web Dependencies
```bash
pip install Flask>=2.3.0 Werkzeug>=2.3.0
```

Or update from requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Before running the web app, you need a trained model:
```bash
python models/train_cnn.py
```

This will create `models/trained_models/baseline_best.keras`.

### 3. Run the Web Server
```bash
python app.py
```

Output:
```
 * Running on http://127.0.0.1:5000
```

### 4. Open in Browser
Visit: **http://127.0.0.1:5000**

## Usage

1. **Upload an Image**
   - Click the upload box or drag & drop an image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP, WebP
   - Max file size: 50MB

2. **Get Results**
   - Click "🔍 Detect" to run inference
   - View prediction: REAL or FAKE
   - Confidence score and probability breakdown

3. **Try Another**
   - Click "Try Another Image" or "Clear" to reset

## API Endpoints

### `POST /api/predict`
Upload an image and get a prediction.

**Request:**
```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:5000/api/predict
```

**Response:**
```json
{
  "success": true,
  "filename": "image.jpg",
  "result": {
    "label": "REAL",
    "confidence": 0.95,
    "real_prob": 0.95,
    "fake_prob": 0.05,
    "raw_prediction": 0.05
  }
}
```

### `GET /api/health`
Check server and model status.

```bash
curl http://127.0.0.1:5000/api/health
```

### `GET /api/info`
Get model information.

```bash
curl http://127.0.0.1:5000/api/info
```

## File Structure
```
.
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web UI (HTML/CSS/JS)
├── uploads/              # Temporary uploaded files (auto-created)
└── models/
    └── trained_models/
        └── baseline_best.keras  # Trained model
```

## Troubleshooting

### Model Not Found
**Error:** `FileNotFoundError: Model not found at models/trained_models/baseline_best.keras`

**Solution:** Train the model first:
```bash
python models/train_cnn.py
```

### Port Already in Use
**Error:** `Address already in use`

**Solution:** Use a different port:
```python
# In app.py, change the last line to:
app.run(debug=True, host='127.0.0.1', port=5001)
```

### Module Not Found (Flask, TensorFlow, etc.)
**Error:** `ModuleNotFoundError: No module named 'flask'`

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

## Performance Notes
- First inference takes ~2-3 seconds (model warmup)
- Subsequent predictions are faster (~1-2 seconds)
- Processes images up to 50MB
- Optimized for GPU inference (if available)

## Deployment

### Local Development
```bash
python app.py
```

### Production
Use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Or Waitress (cross-platform):
```bash
pip install waitress
waitress-serve --port=5000 app:app
```

## Future Enhancements
- [ ] Batch image processing
- [ ] Model explainability (Grad-CAM visualization)
- [ ] Image history/results logging
- [ ] Multiple model support
- [ ] Docker containerization
- [ ] Deploy to cloud (AWS, GCP, Azure)
