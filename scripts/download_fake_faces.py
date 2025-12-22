"""
Download synthetic/fake face images for training data.

Sources:
- ThisPersonDoesNotExist.com (StyleGAN-generated faces)
- AI-generated deepfakes from public sources
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent.parent / 'data' / 'seed' / 'fake'
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_synthetic_faces(count=100):
    """
    Download AI-generated synthetic faces from ThisPersonDoesNotExist.com.
    
    These are NOT deepfakes, but StyleGAN-generated synthetic faces.
    They're useful for training, but ideally you should use real deepfakes
    from deepfake detection datasets (FaceForensics++, DFDC, etc.)
    """
    print(f"\n📥 Downloading {count} synthetic AI-generated faces...")
    print("   Source: https://thistle-playground-images.s3.amazonaws.com/")
    
    api_url = "https://this-person-does-not-exist.com/api?uuid="
    
    successful = 0
    for i in range(count):
        try:
            # Generate a random UUID for variety
            import uuid
            unique_id = str(uuid.uuid4())
            url = f"{api_url}{unique_id}"
            
            dest = DATA_DIR / f"synthetic_{i:05d}.jpg"
            urllib.request.urlretrieve(url, dest)
            successful += 1
            
            if (i + 1) % 10 == 0:
                print(f"   ✓ Downloaded {successful} synthetic faces...")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ⚠ Failed to download synthetic face {i}: {e}")
            continue
    
    print(f"   ✓ Total: {successful} synthetic faces downloaded")
    return successful


def create_fake_faces_guide():
    """
    Provide guidance on obtaining real deepfakes for training.
    """
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║  IMPORTANT: Synthetic Faces vs Real Deepfakes                             ║
╚════════════════════════════════════════════════════════════════════════════╝

The script can download AI-GENERATED synthetic faces (StyleGAN, e.g., from 
ThisPersonDoesNotExist.com), but for a REAL deepfake detector, you need:

REAL DEEPFAKE DATASETS:
─────────────────────

1. FaceForensics++ (Recommended)
   URL: https://github.com/ondyari/FaceForensics
   - 1,000 videos with multiple deepfake methods (Face2Face, FaceSwap, Neural Textures)
   - Setup:
     a. Register at https://docs.google.com/forms/d/e/1FAIpQLSdRibt...
     b. Run their download script
     c. Extract frames using ffmpeg
   - Video guide: https://github.com/ondyari/FaceForensics#download

2. DFDC (Deepfake Detection Challenge)
   URL: https://www.kaggle.com/competitions/deepfake-detection-challenge/data
   - 100K+ video frames (fake and real)
   - Kaggle competition data (requires Kaggle account)
   - Download: kaggle competitions download -c deepfake-detection-challenge

3. CelebDF
   URL: https://github.com/yuezunli/celeb-df
   - 590 original videos + 5,639 deepfake videos
   - High visual quality

QUICK START:
────────────
If you don't want to deal with video, use FRAME-BASED datasets:

Option A: Download pre-extracted frames from FaceForensics++ (via Kaggle):
  https://www.kaggle.com/datasets/rftexastech/faceforensics-deepfake

Option B: Convert your own videos to frames:
  ffmpeg -i video.mp4 -q:v 2 -f image2 frame_%04d.jpg

SYNTHETIC vs REAL:
──────────────────
✓ Synthetic faces (ThisPersonDoesNotExist): Easy to download, but detector 
  will NOT generalize to real deepfakes (different artifacts)

✗ Real deepfakes (FaceForensics++): Harder to obtain, but essential for 
  training a model that works on actual deepfakes

RECOMMENDATION:
───────────────
Split your training data:
  - 40% Real faces (LFW, CelebA, etc.)
  - 60% Real deepfakes (FaceForensics++, DFDC, etc.)

This gives your model better generalization.

╔════════════════════════════════════════════════════════════════════════════╗
"""
    print(guide)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🎬 Fake/Synthetic Face Image Downloader")
    print("="*70)
    
    print("""
⚠️  IMPORTANT NOTE:
This script can download SYNTHETIC AI-generated faces, but for a real
deepfake detector, you need ACTUAL DEEPFAKES from datasets like:
  - FaceForensics++
  - DFDC (Kaggle)
  - CelebDF

See --guide for setup instructions.
""")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--guide":
        create_fake_faces_guide()
    
    elif len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            count = 100
        
        successful = download_synthetic_faces(count)
        
        print(f"\n✓ Downloaded {successful} synthetic faces to {DATA_DIR}")
        print("\n⚠️  Note: These are SYNTHETIC faces, not real deepfakes.")
        print("    For best results, use real deepfakes from FaceForensics++")
        print("    Run: python scripts/download_fake_faces.py --guide")
    
    else:
        # Default: download 100 synthetic faces + show guide
        print("\nDownloading 100 synthetic faces as demo...\n")
        successful = download_synthetic_faces(100)
        
        print("\n" + "="*70)
        create_fake_faces_guide()
        print("="*70)
    
    print("\n" + "="*70)
