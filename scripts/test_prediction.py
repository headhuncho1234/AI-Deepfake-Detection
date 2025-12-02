import numpy as np
from tensorflow import keras
import os

# Load the model
model_path = 'models/trained_models/baseline_best.keras'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

model = keras.models.load_model(model_path)

# Load a test image
img_path = 'data/validation/fake/image_001.jpg'
if not os.path.exists(img_path):
    # Try to find any jpg in the directory
    import glob
    files = glob.glob('data/validation/**/*.jpg', recursive=True)
    if files:
        img_path = files[0]
        print(f"Using found image: {img_path}")
    else:
        print(f"No images found in data/validation")
        exit(1)

img = keras.utils.load_img(img_path, target_size=(224, 224))
img_array = keras.utils.img_to_array(img) # EfficientNet expects [0, 255]
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]
print(f"Raw Prediction Output: {prediction:.4f} (0=Fake, 1=Real)")

if prediction > 0.5:
    verdict = "REAL"
    confidence = prediction * 100
else:
    verdict = "FAKE"
    confidence = (1 - prediction) * 100

print(f"Verdict: {verdict} ({confidence:.2f}% confidence)")

# Expected output (example):
# Prediction: 0.8734
# Verdict: FAKE (87.34% confidence)