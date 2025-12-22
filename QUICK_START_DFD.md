# Quick Start: Download DFD Dataset from Kaggle

## 1. Install Dependencies

```bash
source .venv/bin/activate
pip install kagglehub
```

## 2. Set Up Kaggle API Credentials

### Mac/Linux:
```bash
# Go to https://www.kaggle.com/settings/account
# Click "Create New API Token" (downloads kaggle.json)

# Move the file to the correct location
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Windows:
```
# Move kaggle.json to: C:\Users\<YourUsername>\.kaggle\kaggle.json
```

## 3. Verify Setup

```bash
python scripts/download_dfd_kaggle.py --verify-only
```

Output should be:
```
✓ Kaggle credentials found
✓ Setup verification complete
```

## 4. Download and Organize Dataset

```bash
# This will download ~10GB+ and organize into data/ directory
python scripts/download_dfd_kaggle.py
```

Expected output:
```
Found 7000 real images
Found 7000 fake images

Splitting data (80% train, 20% validation):
  Real: 5600 train, 1400 validation
  Fake: 5600 train, 1400 validation

Copying files...
[████████████████████] 5600/5600
...

✓ Dataset organized successfully!

Training data:
  data/seed/real:  5600 images
  data/seed/fake:  5600 images

Validation data:
  data/validation/real:  1400 images
  data/validation/fake:  1400 images

Total images: 14000
```

## 5. Retrain Model

```bash
python models/train_cnn.py
```

This will now train on **14,000 real images** instead of random noise!

## 6. Test Web Interface

```bash
# Make sure the server is not already running
pkill -f "python app.py" || true

# Start the web server
python app.py

# Open in browser: http://localhost:8000
```

---

## Troubleshooting

### Error: "Kaggle credentials not found"

```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# If not, download it:
# 1. Go to https://www.kaggle.com/settings/account
# 2. Scroll down to "API"
# 3. Click "Create New API Token"
# 4. Extract kaggle.json from the downloaded zip
# 5. Place in ~/.kaggle/
```

### Error: "Dataset not found or access denied"

- Make sure you're logged into Kaggle
- Try manually downloading from: https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset
- Click "Download" and extract to `data/`

### Download is slow

- This is normal! The dataset is ~10GB+
- Download speed depends on your internet connection
- You can cancel and retry with `--verify-only` to skip download next time

### Out of storage space

The dataset requires ~20GB during download and organization:
- Download: ~10GB
- Extracted: ~10GB
- Organized (images copied): ~5GB total in `data/`

---

## Next Steps

1. ✅ Install kagglehub
2. ✅ Set up Kaggle credentials
3. ✅ Download dataset
4. ✅ Retrain model
5. ✅ Test web interface
6. 📊 Evaluate model on test set: `python scripts/evaluate_model.py`
7. 🚀 Deploy to production

---

## Alternative: Manual Download

If kagglehub fails, you can manually download from Kaggle:

1. Go to: https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset
2. Click "Download" (requires Kaggle account)
3. Extract to: `data/`
4. Run: `python scripts/organize_dfd_dataset.py` (if needed)

---

## Expected Performance After Retraining

With 11,200 training images (5,600 real + 5,600 fake):
- **Before (dummy data)**: ~50% accuracy (random guessing)
- **After (real data)**: 85-95% accuracy (depending on model depth)

For best results:
- Use more data (20K+ images)
- Train longer (50+ epochs)
- Use GPU acceleration (if available)
