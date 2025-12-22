# Training Data Guide

## Problem: Why Your Model Performs Poorly

Your model was trained on **random dummy images** (100% noise), not actual face images. This is why it has no predictive power.

To build a working deepfake detector, you need:
- **Real photos**: Actual human faces (not AI-generated)
- **Fake photos**: Real deepfakes (not synthetic AI faces)

## Quick Start: Download Real Data

### 1. Download Real Face Images

```bash
# Option A: Download LFW (Labeled Faces in the Wild) - ~13K images
python scripts/download_real_faces.py
# Then choose option 1

# Option B: From a URL list
echo "https://example.com/face1.jpg" > urls.txt
echo "https://example.com/face2.jpg" >> urls.txt
python scripts/download_real_faces.py --from-urls urls.txt
```

### 2. Download Fake/Deepfake Images

```bash
# Get guide on deepfake datasets
python scripts/download_fake_faces.py --guide

# Download synthetic faces (quick demo, not recommended for production)
python scripts/download_fake_faces.py 100
```

### 3. Retrain the Model

```bash
python models/train_cnn.py
```

---

## Recommended Data Sources

### Real Face Images

| Source | Count | License | Setup |
|--------|-------|---------|-------|
| **LFW** (Labeled Faces in the Wild) | 13,233 | Public | Auto-download (see below) |
| **CelebA** | 202,599 | CC0 (via agreement) | Sign agreement + manual download |
| **FFHQ** (Flickr-Faces-HQ) | 70,000 | CC0 | Manual download (high quality) |
| **VGGFace2** | 3.31M | Academic use | Register + request access |
| **Unsplash** | Unlimited | CC0 | Flickr API / manual download |

**Quick: Use LFW**
```bash
python scripts/download_real_faces.py
# Choose option 1 (auto-downloads LFW)
```

---

### Fake/Deepfake Images

| Dataset | Count | Methods | Access | Setup |
|---------|-------|---------|--------|-------|
| **FaceForensics++** | 1,000 videos | Face2Face, FaceSwap, NeuralTextures, DeeperForensics | Public | Form submission + download script |
| **DFDC** | 100K+ frames | Multiple methods | Kaggle | Kaggle API or manual download |
| **CelebDF** | 5,639 videos | DeepfaceLab | Public | GitHub repo + download |
| **DeepFaceLab** | ~10K videos | Community-sourced | Public | GitHub repo |

**Recommended: FaceForensics++**

1. Visit: https://github.com/ondyari/FaceForensics
2. Fill out the access form
3. Follow their download script
4. Extract frames:
```bash
# Convert videos to image frames (example)
ffmpeg -i video.mp4 -q:v 2 -f image2 frame_%04d.jpg
```
5. Move frames to `data/seed/fake/`

---

## Data Organization

After downloading, organize as follows:

```
data/
├── seed/                     # Training set
│   ├── real/                 # Real face images
│   │   ├── person_001.jpg
│   │   ├── person_002.jpg
│   │   └── ...
│   └── fake/                 # Deepfake images
│       ├── deepfake_001.jpg
│       ├── deepfake_002.jpg
│       └── ...
│
├── validation/               # Validation set (20% of seed)
│   ├── real/
│   └── fake/
│
└── pool/                     # Optional: Active learning pool
    ├── real/
    └── fake/
```

**Script to organize:**
```bash
# Split downloaded images into train/val automatically
python scripts/split_dataset.py --real-dir /path/to/real/images --fake-dir /path/to/fake/images
```

---

## Recommended Workflow

### Phase 1: Quick Test (30 min)
```bash
# Download ~100 real faces and use synthetic faces for testing
python scripts/download_real_faces.py    # Select LFW
python scripts/download_fake_faces.py 100

# Quick retrain
python models/train_cnn.py
```

### Phase 2: Production Model (hours/days)
```bash
# Download FaceForensics++ (~50GB)
# Instructions: https://github.com/ondyari/FaceForensics

# Or use DFDC from Kaggle
kaggle competitions download -c deepfake-detection-challenge
# Extract and organize

# Retrain with full data
python models/train_cnn.py
```

---

## Troubleshooting

### "Object of type float32 is not JSON serializable" (Web Interface)
Fixed in latest `app.py`. Restart the server:
```bash
pkill -f "python app.py"
python app.py
```

### Downloads are slow
- Use a download manager (e.g., `aria2c`)
- Try alternative mirrors if available
- Consider pre-downloaded datasets from Kaggle

### Not enough storage
- LFW: ~200 MB
- CelebA: ~1.3 GB
- FaceForensics++: ~50 GB
- DFDC: ~400 GB

Choose datasets based on your storage budget.

---

## API Reference: Download Scripts

### `scripts/download_real_faces.py`

```bash
# Interactive menu
python scripts/download_real_faces.py

# From URLs file
python scripts/download_real_faces.py --from-urls urls.txt
```

### `scripts/download_fake_faces.py`

```bash
# Show setup guide
python scripts/download_fake_faces.py --guide

# Download N synthetic faces
python scripts/download_fake_faces.py 50
```

---

## Recommended Models & Parameters

With proper training data, adjust `models/train_cnn.py`:

```python
# For large datasets (10K+ images per class)
epochs = 50
batch_size = 128
learning_rate = 0.001

# For medium datasets (1K-10K images per class)
epochs = 30
batch_size = 64
learning_rate = 0.0005

# For small datasets (<1K images per class)
epochs = 20
batch_size = 32
learning_rate = 0.0001
use_augmentation = True  # More data augmentation
```

---

## Next Steps

1. **Download real data** using scripts above
2. **Organize** data into `data/seed/{real,fake}/` structure
3. **Retrain** model: `python models/train_cnn.py`
4. **Evaluate** on test set: `python scripts/evaluate_model.py`
5. **Deploy** web interface: `python app.py`

---

## References

- FaceForensics++: https://github.com/ondyari/FaceForensics
- DFDC: https://www.kaggle.com/competitions/deepfake-detection-challenge
- LFW: http://vis-www.cs.umass.edu/lfw/
- CelebDF: https://github.com/yuezunli/celeb-df
- Deepfake Detection Review: https://arxiv.org/abs/2001.00686
