"""
Download smaller, curated deepfake datasets suitable for local development.

Options:
1. FSGAN-generated deepfakes (~500MB)
2. YouTube Deepfake Detection (~1GB)
3. Deepfake TIMIT (~2GB, ~3k videos)
4. Manual curation from public sources
"""

import os
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, dest_path, chunk_size=8192):
    """Download file with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"  Downloaded: {downloaded/1e6:.1f}MB / {total_size/1e6:.1f}MB ({pct:.1f}%)", end='\r')
        
        print()
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def option_1_fsgan():
    """
    FSGAN-generated deepfakes (~500MB)
    Small, synthetic deepfakes good for quick testing.
    """
    print("\n" + "="*70)
    print("Option 1: FSGAN Deepfakes (~500MB)")
    print("="*70)
    print("""
FSGAN is a face-swapping method that generates synthetic deepfakes.
- Size: ~500MB
- Images: ~5K deepfake + real pairs
- Quality: Good for testing, not real-world deepfakes
- Download time: ~10-30 minutes

Note: These are synthetic deepfakes, not face-reenactment videos.
""")
    print("Source: https://github.com/YuvalNirkin/fsgan")
    print("Pre-built dataset: https://github.com/YuvalNirkin/fsgan/releases")


def option_2_youtube():
    """
    YouTube Deepfake Detection (~1GB)
    Real deepfakes extracted from YouTube.
    """
    print("\n" + "="*70)
    print("Option 2: YouTube Deepfake Detection (~1GB)")
    print("="*70)
    print("""
Real deepfakes extracted from YouTube videos.
- Size: ~1GB
- Content: Face-swap and lip-sync deepfakes
- Quality: Real-world deepfakes
- Download time: ~30-60 minutes

Setup:
  1. Visit: https://www.kaggle.com/datasets/aainz/youtube-deepfake-detection
  2. Download manually or use Kaggle API: kaggle datasets download -d aainz/youtube-deepfake-detection
  3. Extract to data/ directory
""")


def option_3_timit():
    """
    Deepfake TIMIT (~2GB)
    Controlled lab-recorded deepfakes.
    """
    print("\n" + "="*70)
    print("Option 3: Deepfake TIMIT (~2GB)")
    print("="*70)
    print("""
Lab-recorded controlled deepfakes from TIMIT dataset.
- Size: ~2GB
- Videos: 3,200 (640 unique speakers)
- Quality: High-quality, controlled environment
- Download time: ~60-90 minutes

Setup:
  1. Register at: http://www.deepfaketimit.org/
  2. Download dataset
  3. Extract frames to data/
  
For frame extraction:
  ffmpeg -i video.mp4 -q:v 2 -f image2 frame_%04d.jpg
""")


def option_4_celeba_subset():
    """
    CelebA + Synthetic Deepfakes (~3GB)
    Real CelebA images + manually created deepfakes.
    """
    print("\n" + "="*70)
    print("Option 4: CelebA Subset + DIY Deepfakes (~3GB)")
    print("="*70)
    print("""
Use real CelebA images + create your own deepfakes.

Real images (CelebA):
  - Size: ~1.3GB
  - Images: ~200K celebrity faces
  - Setup: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  - Alternative (aligned): https://www.kaggle.com/jessicali9530/celeba-dataset

Deepfakes (create yourself):
  - Use DeepfaceLab: https://github.com/iperov/DeepFaceLab
  - Use first-order-motion: https://github.com/aliaksandrul/first-order-motion-model
  - Time: 2-4 hours per deepfake video
""")


def option_5_quick_demo():
    """
    Quick demo dataset (~100MB)
    Perfect for testing model code without waiting.
    """
    print("\n" + "="*70)
    print("Option 5: Quick Demo Dataset (~100MB)")
    print("="*70)
    print("""
Pre-curated small dataset for rapid testing.
- Size: ~100MB
- Images: 500 real + 500 fake
- Time to download: ~5 minutes
- Time to train: ~5 minutes

Best for:
  ✓ Testing code changes
  ✓ Quick iterations
  ✓ Verifying pipeline works

Not suitable for:
  ✗ Production models
  ✗ Accuracy benchmarks
""")
    
    # Could implement automated download here
    print("\nTo create this yourself:")
    print("  1. Download 500 real faces (LFW): python scripts/download_real_faces.py")
    print("  2. Download 500 synthetic faces: python scripts/download_fake_faces.py 500")
    print("  3. Organize into data/seed/{real,fake}/")
    print("  4. Train: python models/train_cnn.py")


def option_6_kaggle_alternatives():
    """
    Other Kaggle datasets (1-3GB each)
    """
    print("\n" + "="*70)
    print("Option 6: Other Kaggle Datasets")
    print("="*70)
    print("""
Smaller Kaggle datasets (~1-3GB each):

1. Deepfake Detection Challenge (DFDC) - Smaller Subset
   https://www.kaggle.com/datasets/abhikjha/deepfake-detection-challenge-small

2. Face Swap Detection
   https://www.kaggle.com/datasets/aainz/face-swap-detection

3. Synthetic Faces (StyleGAN)
   https://www.kaggle.com/datasets/arnaudmout/thispersondoesnotexist-faces

Download with Kaggle API:
  kaggle datasets download -d abhikjha/deepfake-detection-challenge-small
  unzip deepfake-detection-challenge-small.zip -d data/
""")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("🎬 Lightweight Deepfake Dataset Options")
    print("="*80)
    print("""
The full DFD dataset is too large (10GB+). Here are smaller alternatives:
""")
    
    options = {
        "1": ("FSGAN (~500MB, synthetic)", option_1_fsgan),
        "2": ("YouTube Deepfakes (~1GB, real)", option_2_youtube),
        "3": ("Deepfake TIMIT (~2GB, controlled)", option_3_timit),
        "4": ("CelebA Subset + DIY (~3GB)", option_4_celeba_subset),
        "5": ("Quick Demo (~100MB)", option_5_quick_demo),
        "6": ("Other Kaggle Datasets (~1-3GB)", option_6_kaggle_alternatives),
        "all": ("Show all options", None),
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in options:
        choice = sys.argv[1]
        if choice == "all":
            for opt, (name, func) in options.items():
                if opt != "all" and func:
                    func()
        else:
            name, func = options[choice]
            if func:
                func()
    else:
        print("\nAvailable options:\n")
        for opt, (name, _) in options.items():
            if opt != "all":
                print(f"  {opt}. {name}")
        print(f"  all. Show all options")
        
        print("\nUsage:")
        print("  python scripts/download_lightweight_datasets.py <option>")
        print("\nExamples:")
        print("  python scripts/download_lightweight_datasets.py 1  # FSGAN")
        print("  python scripts/download_lightweight_datasets.py 5  # Quick demo")
        print("  python scripts/download_lightweight_datasets.py all  # All info")
    
    print("\n" + "="*80)
    print("RECOMMENDATION for your use case:")
    print("="*80)
    print("""
Best option: Option 5 (Quick Demo) → Option 2 (YouTube Deepfakes)

Workflow:
  1. Start with Quick Demo (~100MB, 5 min download)
     - Test model code works
     - Quick iteration cycle
  
  2. When ready, upgrade to YouTube Deepfakes (~1GB, 30-60 min)
     - Better real-world performance
     - Still manageable size
  
  3. Future: Deepfake TIMIT (~2GB) for production quality

""")
