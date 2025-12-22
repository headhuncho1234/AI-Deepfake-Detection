"""
Download and organize the Deep Fake Detection (DFD) dataset from Kaggle.

This script uses kagglehub to download the deepfake dataset and organizes it
into the proper structure for model training.

Setup:
  1. Install kagglehub: pip install kagglehub
  2. Setup Kaggle credentials: https://www.kaggle.com/settings/account
     - Generate API token (downloads kaggle.json)
     - Place in ~/.kaggle/kaggle.json (or set KAGGLE_CONFIG_DIR)
  3. Run this script
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

# Try to import kagglehub
try:
    import kagglehub
except ImportError:
    print("❌ kagglehub not installed!")
    print("   Install with: pip install kagglehub")
    sys.exit(1)


def download_dfd_dataset():
    """Download the DFD (Deep Fake Detection) dataset from Kaggle."""
    print("\n" + "="*70)
    print("📥 Downloading Deep Fake Detection Dataset from Kaggle")
    print("="*70)
    
    dataset_name = "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset"
    
    print(f"\nDataset: {dataset_name}")
    print("Downloading... (this may take several minutes/hours)")
    print("Note: Dataset is ~10GB+, so be patient!\n")
    
    try:
        path = kagglehub.dataset_download(dataset_name)
        print(f"\n✓ Dataset downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("\nMake sure:")
        print("  1. kagglehub is installed: pip install kagglehub")
        print("  2. Kaggle API credentials are set up:")
        print("     - Visit: https://www.kaggle.com/settings/account")
        print("     - Click 'Create New API Token' (downloads kaggle.json)")
        print("     - Place it in ~/.kaggle/kaggle.json")
        return None


def organize_dfd_dataset(dataset_path, output_dir=None):
    """
    Organize the downloaded DFD dataset into training/validation structure.
    
    Expected DFD structure:
        dataset/
        ├── REAL/
        │   └── *.png
        └── FAKE/
            └── *.png
    """
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("📁 Organizing dataset into training/validation structure")
    print("="*70)
    
    dataset_path = Path(dataset_path)
    
    # Find REAL and FAKE directories in the downloaded dataset
    real_src = dataset_path / "REAL"
    fake_src = dataset_path / "FAKE"
    
    if not real_src.exists() or not fake_src.exists():
        # Try to find them recursively
        for root, dirs, files in os.walk(dataset_path):
            for d in dirs:
                if d.upper() == "REAL":
                    real_src = Path(root) / d
                if d.upper() == "FAKE":
                    fake_src = Path(root) / d
    
    if not real_src.exists():
        print(f"❌ Could not find REAL directory in {dataset_path}")
        return False
    
    if not fake_src.exists():
        print(f"❌ Could not find FAKE directory in {dataset_path}")
        return False
    
    print(f"Found REAL: {real_src}")
    print(f"Found FAKE: {fake_src}")
    
    # Create destination directories
    seed_real = output_dir / "seed" / "real"
    seed_fake = output_dir / "seed" / "fake"
    val_real = output_dir / "validation" / "real"
    val_fake = output_dir / "validation" / "fake"
    
    for d in [seed_real, seed_fake, val_real, val_fake]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    real_files = list(real_src.glob("*.png")) + list(real_src.glob("*.jpg"))
    fake_files = list(fake_src.glob("*.png")) + list(fake_src.glob("*.jpg"))
    
    print(f"\nFound {len(real_files)} real images")
    print(f"Found {len(fake_files)} fake images")
    
    if len(real_files) == 0 and len(fake_files) == 0:
        print("❌ No images found! Check dataset structure.")
        return False
    
    # Split into train/validation (80/20)
    split_ratio = 0.8
    real_split = int(len(real_files) * split_ratio)
    fake_split = int(len(fake_files) * split_ratio)
    
    real_train = real_files[:real_split]
    real_val = real_files[real_split:]
    fake_train = fake_files[:fake_split]
    fake_val = fake_files[fake_split:]
    
    print(f"\nSplitting data (80% train, 20% validation):")
    print(f"  Real: {len(real_train)} train, {len(real_val)} validation")
    print(f"  Fake: {len(fake_train)} train, {len(fake_val)} validation")
    
    # Copy files
    print("\nCopying files...")
    
    for files, dest in [
        (real_train, seed_real),
        (real_val, val_real),
        (fake_train, seed_fake),
        (fake_val, val_fake),
    ]:
        for src_file in tqdm(files, desc=f"→ {dest.name}"):
            try:
                dst_file = dest / src_file.name
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"  ⚠ Failed to copy {src_file}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ Dataset organized successfully!")
    print("="*70)
    
    print(f"\nTraining data:")
    print(f"  data/seed/real:  {len(list(seed_real.glob('*')))} images")
    print(f"  data/seed/fake:  {len(list(seed_fake.glob('*')))} images")
    
    print(f"\nValidation data:")
    print(f"  data/validation/real:  {len(list(val_real.glob('*')))} images")
    print(f"  data/validation/fake:  {len(list(val_fake.glob('*')))} images")
    
    total = (
        len(list(seed_real.glob('*'))) +
        len(list(seed_fake.glob('*'))) +
        len(list(val_real.glob('*'))) +
        len(list(val_fake.glob('*')))
    )
    
    print(f"\nTotal images: {total}")
    print(f"Location: {output_dir}")
    
    return True


def verify_kaggle_setup():
    """Verify Kaggle API credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\n❌ Kaggle credentials not found!")
        print("\nSetup instructions:")
        print("  1. Go to: https://www.kaggle.com/settings/account")
        print("  2. Scroll down and click 'Create New API Token'")
        print("  3. This downloads kaggle.json")
        print("  4. Move it to: ~/.kaggle/kaggle.json")
        print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✓ Kaggle credentials found")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and organize DFD dataset from Kaggle"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify Kaggle setup, don't download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for organized data (default: data/)"
    )
    
    args = parser.parse_args()
    
    # Verify setup
    if not verify_kaggle_setup():
        if not args.verify_only:
            sys.exit(1)
    
    if args.verify_only:
        print("\n✓ Setup verification complete")
        sys.exit(0)
    
    # Download dataset
    dataset_path = download_dfd_dataset()
    
    if dataset_path is None:
        sys.exit(1)
    
    # Organize dataset
    success = organize_dfd_dataset(dataset_path, args.output_dir)
    
    if success:
        print("\n" + "="*70)
        print("🎉 Ready to train!")
        print("="*70)
        print("\nNext step:")
        print("  python models/train_cnn.py")
        print("\nOr run the web interface:")
        print("  python app.py")
        print("="*70)
    else:
        sys.exit(1)
