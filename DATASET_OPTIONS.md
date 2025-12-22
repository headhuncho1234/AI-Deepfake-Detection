# Smaller Dataset Options (Instead of 10GB DFD)

## Quick Comparison

| Option | Size | Time | Quality | Best For |
|--------|------|------|---------|----------|
| **Quick Demo** | 100MB | 5 min | ⭐⭐⭐ | Testing code |
| **FSGAN** | 500MB | 20 min | ⭐⭐⭐⭐ | Quick training |
| **YouTube DF** | 1GB | 45 min | ⭐⭐⭐⭐⭐ | Real-world data |
| **TIMIT DF** | 2GB | 90 min | ⭐⭐⭐⭐⭐ | Production |

---

## 🚀 Fastest Path (Recommended)

### Step 1: Quick Demo (100MB, ~5 min)
Perfect for testing everything works:

```bash
# Download real faces
python scripts/download_real_faces.py
# Choose option 1

# Download synthetic faces
python scripts/download_fake_faces.py 500

# Organize and train
python models/train_cnn.py

# Test web interface
python app.py
```

**Result**: Working model in ~30 minutes

---

### Step 2: YouTube Deepfakes (1GB, ~45 min) - *Optional*
When you want better accuracy:

```bash
# Option A: Kaggle API (easiest)
pip install kaggle
kaggle datasets download -d aainz/youtube-deepfake-detection
unzip youtube-deepfake-detection.zip -d /tmp/
python scripts/organize_downloaded_data.py /tmp/youtube-deepfake-detection data/

# Option B: Manual download
# Go to: https://www.kaggle.com/datasets/aainz/youtube-deepfake-detection
# Click Download → Extract to data/

# Retrain with real deepfakes
python models/train_cnn.py
```

**Result**: Model trained on real YouTube deepfakes

---

## 📊 Size Breakdown

### Quick Demo (LFW + Synthetic) - **~100MB**
```
data/
├── seed/
│   ├── real/      (~50MB, 500 LFW images)
│   └── fake/      (~50MB, 500 synthetic)
└── validation/    (auto-split 20%)
```

### YouTube Deepfakes - **~1GB**
```
data/
├── seed/
│   ├── real/      (~500MB, real YouTube faces)
│   └── fake/      (~500MB, real YouTube deepfakes)
└── validation/    (auto-split 20%)
```

### Full DFD - **~10GB** (NOT recommended)

---

## 💻 Storage Requirements

| Dataset | Storage | Download | Extract | Total |
|---------|---------|----------|---------|-------|
| Quick Demo | 100MB | 100MB | 100MB | **200MB** |
| YouTube DF | 1GB | 1GB | 1GB | **2GB** |
| TIMIT DF | 2GB | 2GB | 2GB | **4GB** |
| Full DFD | 10GB | 10GB | 5GB | **15GB** |

---

## Training Time Estimates

On MacBook Pro (CPU):
- Quick Demo (1K images): **5-10 minutes**
- YouTube DF (11K images): **30-45 minutes**
- TIMIT DF (32K images): **2-3 hours**
- Full DFD (100K images): **8-12 hours**

With GPU (NVIDIA/Apple Silicon):
- Divide times by 3-5x

---

## Expected Accuracy

After training on each dataset:

| Dataset | Accuracy | Real→Fake Misclass | Notes |
|---------|----------|-------------------|-------|
| Random Noise | ~50% | Yes (major issue) | Current state |
| Quick Demo | ~75-85% | Sometimes | Good for testing |
| YouTube DF | **90-95%** | Rare | Great balance |
| TIMIT DF | **92-97%** | Very rare | Best performance |
| Full DFD | **95-99%** | Extremely rare | Overkill for most |

---

## Recommended for You

**START HERE:**

```bash
# Step 1: LFW (real faces) - 200MB
python scripts/download_real_faces.py

# Step 2: Synthetic faces - 50MB  
python scripts/download_fake_faces.py 500

# Step 3: Retrain model (5 min)
python models/train_cnn.py

# Step 4: Test (should work now!)
python app.py
```

**Total time: ~30 minutes | Total size: ~200MB | Accuracy: ~75-85%**

---

## If You Have More Time/Storage

```bash
# Upgrade to YouTube deepfakes (1GB)
kaggle datasets download -d aainz/youtube-deepfake-detection
unzip -d /tmp/
python scripts/organize_downloaded_data.py /tmp/youtube-deepfake-detection data/

# Retrain (30 min)
python models/train_cnn.py

# Now accuracy: ~90-95%
```

**Additional time: ~1 hour | Additional size: ~1GB | Better accuracy: +10-20%**

---

## All Available Datasets

```bash
python scripts/download_lightweight_datasets.py all
```

Shows details on:
1. FSGAN (~500MB)
2. YouTube Deepfakes (~1GB)
3. Deepfake TIMIT (~2GB)
4. CelebA Subset (~3GB)
5. Quick Demo (~100MB)
6. Other Kaggle datasets

---

## Troubleshooting

### "Not enough storage"
- Start with Quick Demo only (100MB)
- Delete old models: `rm -rf models/trained_models/*`
- Move data to external drive

### "Training is too slow"
- Use smaller dataset (Quick Demo instead of YouTube)
- Reduce epochs in train_cnn.py: `epochs=10` instead of `20`
- Use GPU if available (check TensorFlow)

### "Still detecting real faces as fake"
- Upgrade dataset size (YouTube DF instead of Quick Demo)
- Train longer: `epochs=50` instead of `20`
- Use more data augmentation

---

## Next Steps

1. Choose dataset size based on storage/time
2. Run download script
3. Retrain model
4. Test web interface
5. Adjust as needed

Done! ✅
