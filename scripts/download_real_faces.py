"""
Download real face images for training data.

Sources:
- Flickr Faces Database (70K+ labeled faces, Creative Commons)
- LFW (Labeled Faces in the Wild) - 13K images
- CelebA subset - 200K+ celebrity images
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm
import shutil

# Create directories
DATA_DIR = Path(__file__).parent.parent / 'data' / 'seed' / 'real'
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_lfw_faces():
    """
    Download Labeled Faces in the Wild (LFW) dataset.
    ~13K high-quality real face images.
    
    Source: http://vis-www.cs.umass.edu/lfw/
    """
    print("\n📥 Downloading LFW (Labeled Faces in the Wild)...")
    print("   This is a large dataset (~200MB). It may take several minutes.")
    
    # LFW tar.gz
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    tar_path = DATA_DIR.parent.parent / "lfw.tgz"
    extract_path = DATA_DIR.parent.parent / "lfw"
    
    try:
        # Download
        print(f"   Downloading from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"   ✓ Downloaded {tar_path.stat().st_size / 1e6:.1f} MB")
        
        # Extract
        print("   Extracting...")
        shutil.unpack_archive(tar_path, extract_path.parent)
        print(f"   ✓ Extracted to {extract_path}")
        
        # Copy images to data/seed/real
        count = 0
        for jpg_file in extract_path.rglob("*.jpg"):
            dest = DATA_DIR / f"lfw_{count:05d}.jpg"
            shutil.copy(jpg_file, dest)
            count += 1
            if count % 100 == 0:
                print(f"   ✓ Copied {count} images...")
        
        print(f"   ✓ Total: {count} real face images copied to {DATA_DIR}")
        
        # Cleanup
        shutil.rmtree(extract_path, ignore_errors=True)
        tar_path.unlink(missing_ok=True)
        
        return count
        
    except Exception as e:
        print(f"   ✗ Error downloading LFW: {e}")
        return 0


def download_celeba_subset():
    """
    Download CelebA subset (200K+ celebrity faces).
    
    Note: Full CelebA requires signing an agreement.
    This uses the aligned & cropped version from Kaggle or alternative sources.
    """
    print("\n📥 Downloading CelebA Subset...")
    print("   Note: For full CelebA, visit https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("         and follow their sign-up process.")
    
    # Alternative: Use a pre-cropped, smaller version
    url = "https://github.com/NVlabs/ffhq-dataset/blob/main/README.md"
    print(f"   → For high-quality 70K images, use FFHQ: https://github.com/NVlabs/ffhq-dataset")
    print(f"   → Download FFHQ images and extract to: {DATA_DIR}")


def download_from_list(urls_file):
    """
    Download images from a text file with one URL per line.
    
    Usage: Create urls.txt with image URLs, then call this function.
    """
    if not Path(urls_file).exists():
        print(f"   File not found: {urls_file}")
        return 0
    
    count = 0
    with open(urls_file, 'r') as f:
        urls = f.readlines()
    
    for idx, url in enumerate(tqdm(urls, desc="Downloading images")):
        url = url.strip()
        if not url:
            continue
        
        try:
            ext = url.split('.')[-1].split('?')[0]  # Get extension
            if ext not in ['jpg', 'jpeg', 'png', 'gif']:
                ext = 'jpg'
            
            dest = DATA_DIR / f"custom_{count:05d}.{ext}"
            urllib.request.urlretrieve(url, dest)
            count += 1
            
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"   ✗ Failed to download {url}: {e}")
            continue
    
    print(f"   ✓ Downloaded {count} images from {urls_file}")
    return count


def create_urls_template():
    """Create a template URLs file for manual image sources."""
    template_file = DATA_DIR.parent.parent / "urls_real_faces.txt"
    
    if not template_file.exists():
        with open(template_file, 'w') as f:
            f.write("""# Add real face image URLs here (one per line)
# Example sources:
# - Flickr API: search for "face" with CC license
# - Unsplash API: https://unsplash.com/
# - Pexels API: https://www.pexels.com/
# - Open image datasets

# Example URLs (replace with actual image URLs):
# https://example.com/face1.jpg
# https://example.com/face2.jpg
""")
        print(f"✓ Created template: {template_file}")
        print("  Edit this file and add real face image URLs, then run:")
        print(f"  python scripts/download_real_faces.py --from-urls {template_file}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🎬 Real Face Image Downloader")
    print("="*70)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--from-urls" and len(sys.argv) > 2:
        # Download from URL list
        urls_file = sys.argv[2]
        download_from_list(urls_file)
    
    else:
        # Default: download LFW
        print("\nOptions:")
        print("  1. Download LFW (Labeled Faces in the Wild) ~13K images")
        print("  2. Learn about CelebA and FFHQ")
        print("  3. Create URLs template (for manual sources)")
        print("  4. Download from URLs file")
        
        choice = input("\nChoose option (1-4, or press Enter for 1): ").strip()
        
        if choice == "1" or choice == "":
            count = download_lfw_faces()
            if count > 0:
                print(f"\n✓ Successfully downloaded {count} real face images!")
                print(f"  Location: {DATA_DIR}")
                print("\n  Next step: Retrain the model with real data")
                print(f"  python models/train_cnn.py")
        
        elif choice == "2":
            download_celeba_subset()
        
        elif choice == "3":
            create_urls_template()
        
        elif choice == "4":
            urls_file = input("Enter path to URLs file: ").strip()
            download_from_list(urls_file)
        
        else:
            print("Invalid choice")
    
    print("\n" + "="*70)
