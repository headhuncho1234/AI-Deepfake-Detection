"""
Organize downloaded images into training/validation structure.
Handles various directory layouts and creates proper split.
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def organize_images(output_dir=None):
    """
    Organize all downloaded images into:
    - data/seed/real/
    - data/seed/fake/
    - data/validation/real/
    - data/validation/fake/
    """
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    seed_real = output_dir / "seed" / "real"
    seed_fake = output_dir / "seed" / "fake"
    val_real = output_dir / "validation" / "real"
    val_fake = output_dir / "validation" / "fake"
    
    for d in [seed_real, seed_fake, val_real, val_fake]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Collect all images that were downloaded
    print("\n📁 Collecting downloaded images...")
    
    # Real images (from download scripts)
    real_images = []
    real_sources = [seed_real]  # Images already here
    
    for source in real_sources:
        if source.exists():
            real_images.extend(list(source.glob("*.jpg")) + list(source.glob("*.png")))
    
    # Fake images (from synthetic download)
    fake_images = []
    fake_sources = [seed_fake]  # Images already here
    
    for source in fake_sources:
        if source.exists():
            fake_images.extend(list(source.glob("*.jpg")) + list(source.glob("*.png")))
    
    print(f"Found {len(real_images)} real images")
    print(f"Found {len(fake_images)} fake images")
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("\n⚠️  Not all images downloaded yet. Try again in a moment.")
        return False
    
    # Shuffle and split
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    split = 0.8
    real_split = int(len(real_images) * split)
    fake_split = int(len(fake_images) * split)
    
    real_train = real_images[:real_split]
    real_val = real_images[real_split:]
    fake_train = fake_images[:fake_split]
    fake_val = fake_images[fake_split:]
    
    print(f"\nSplitting 80% train / 20% validation:")
    print(f"  Real: {len(real_train)} train, {len(real_val)} validation")
    print(f"  Fake: {len(fake_train)} train, {len(fake_val)} validation")
    
    # Move/organize files
    print("\nOrganizing files...")
    
    # Real train - keep in seed/real (already there)
    for f in tqdm(real_train, desc="Real train"):
        if f.parent != seed_real:
            shutil.move(str(f), seed_real / f.name)
    
    # Real validation
    for f in tqdm(real_val, desc="Real validation"):
        shutil.move(str(f), val_real / f.name)
    
    # Fake train - keep in seed/fake (already there)
    for f in tqdm(fake_train, desc="Fake train"):
        if f.parent != seed_fake:
            shutil.move(str(f), seed_fake / f.name)
    
    # Fake validation
    for f in tqdm(fake_val, desc="Fake validation"):
        shutil.move(str(f), val_fake / f.name)
    
    # Summary
    print("\n" + "="*70)
    print("✓ Dataset organized successfully!")
    print("="*70)
    
    train_real_count = len(list(seed_real.glob("*")))
    train_fake_count = len(list(seed_fake.glob("*")))
    val_real_count = len(list(val_real.glob("*")))
    val_fake_count = len(list(val_fake.glob("*")))
    
    print(f"\nTraining set:")
    print(f"  Real:  {train_real_count} images in data/seed/real/")
    print(f"  Fake:  {train_fake_count} images in data/seed/fake/")
    
    print(f"\nValidation set:")
    print(f"  Real:  {val_real_count} images in data/validation/real/")
    print(f"  Fake:  {val_fake_count} images in data/validation/fake/")
    
    total = train_real_count + train_fake_count + val_real_count + val_fake_count
    print(f"\nTotal: {total} images")
    
    return True


if __name__ == "__main__":
    organize_images()
