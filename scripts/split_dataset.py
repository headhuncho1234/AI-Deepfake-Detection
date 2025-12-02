"""
split_dataset.py

Stratified dataset splitting for deepfake detection with RL active learning.

Dataset: 161,001 images (81,000 real, 80,001 fake)
Strategy: Stratified sampling across 4 fake image quartiles
Output: seed/, pool/, validation/ directories with balanced splits
"""

import os
import shutil
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# Dataset paths
FAKE_DIR = 'data/raw/fake'
REAL_DIR = 'data/raw/real'
OUTPUT_DIR = 'data'

# Split sizes
SEED_SIZE_TOTAL = 8050        # 4,025 real + 4,025 fake
VALIDATION_SIZE_TOTAL = 700   # 350 real + 350 fake
# Pool will be: everything else (152,251 images)

# Quartile configuration for fake images
NUM_QUARTILES = 4
TOTAL_FAKE_IMAGES = 80001

# Calculate quartile boundaries
QUARTILE_SIZE = TOTAL_FAKE_IMAGES // NUM_QUARTILES  # 20,000

FAKE_QUARTILES = [
    (0, 20000, "Q1: Images 0-20,000"),
    (20000, 40000, "Q2: Images 20,000-40,000"),
    (40000, 60000, "Q3: Images 40,000-60,000"),
    (60000, 80001, "Q4: Images 60,000-80,001"),
]

# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def setup_output_directories(output_dir):
    """Create output directory structure"""
    splits = ['seed', 'pool', 'validation']
    labels = ['real', 'fake']
    
    for split in splits:
        for label in labels:
            path = Path(output_dir) / split / label
            path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created output directories in: {output_dir}")

def get_sorted_images(directory):
    """Get sorted list of image paths"""
    directory = Path(directory)
    
    # Try multiple extensions
    images = (
        list(directory.glob('*.jpg')) + 
        list(directory.glob('*.jpeg')) + 
        list(directory.glob('*.png'))
    )
    
    # Sort by filename (important for sequential quartiles)
    images = sorted(images, key=lambda x: x.name)
    
    return images

def save_split_metadata(output_dir, metadata):
    """Save split configuration and statistics"""
    metadata_path = Path(output_dir) / 'split_metadata.json'
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata to: {metadata_path}")

# ══════════════════════════════════════════════════════════════
# MAIN SPLITTING FUNCTION
# ══════════════════════════════════════════════════════════════

def stratified_split_dataset():
    """
    Main function to split dataset with stratified sampling
    """
    
    print("=" * 70)
    print("DEEPFAKE DATASET STRATIFIED SPLIT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {RANDOM_SEED}\n")
    
    # Set random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # ══════════════════════════════════════════════════════════
    # STEP 1: LOAD IMAGES
    # ══════════════════════════════════════════════════════════
    
    print("📂 STEP 1: Loading images...")
    
    fake_images = get_sorted_images(FAKE_DIR)
    real_images = get_sorted_images(REAL_DIR)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Fake images: {len(fake_images):,}")
    print(f"   Real images: {len(real_images):,}")
    print(f"   Total: {len(fake_images) + len(real_images):,}")
    
    if len(fake_images) == 0 or len(real_images) == 0:
        raise ValueError("❌ No images found! Check your data paths.")
    
    if len(fake_images) != TOTAL_FAKE_IMAGES:
        print(f"\n⚠️  WARNING: Expected {TOTAL_FAKE_IMAGES:,} fake images, found {len(fake_images):,}")
        print(f"   Adjusting quartile boundaries...")
        
        # Recalculate quartiles based on actual count
        actual_quartile_size = len(fake_images) // NUM_QUARTILES
        global FAKE_QUARTILES
        FAKE_QUARTILES = [
            (i * actual_quartile_size, 
             (i + 1) * actual_quartile_size if i < NUM_QUARTILES - 1 else len(fake_images),
             f"Q{i+1}: Images {i * actual_quartile_size:,}-{(i + 1) * actual_quartile_size if i < NUM_QUARTILES - 1 else len(fake_images):,}")
            for i in range(NUM_QUARTILES)
        ]
    
    # ══════════════════════════════════════════════════════════
    # STEP 2: CALCULATE SPLIT SIZES
    # ══════════════════════════════════════════════════════════
    
    print(f"\n📐 STEP 2: Calculating split sizes...")
    
    # Seed
    seed_real_count = SEED_SIZE_TOTAL // 2
    seed_fake_count = SEED_SIZE_TOTAL - seed_real_count
    seed_fake_per_quartile = seed_fake_count // NUM_QUARTILES
    
    # Validation
    val_real_count = VALIDATION_SIZE_TOTAL // 2
    val_fake_count = VALIDATION_SIZE_TOTAL - val_real_count
    val_fake_per_quartile = val_fake_count // NUM_QUARTILES
    
    print(f"\n📋 Split Configuration:")
    print(f"\n   SEED ({SEED_SIZE_TOTAL:,} images):")
    print(f"      Real: {seed_real_count:,}")
    print(f"      Fake: {seed_fake_count:,}")
    print(f"         → {seed_fake_per_quartile} per quartile × {NUM_QUARTILES} quartiles")
    
    print(f"\n   VALIDATION ({VALIDATION_SIZE_TOTAL:,} images):")
    print(f"      Real: {val_real_count:,}")
    print(f"      Fake: {val_fake_count:,}")
    print(f"         → {val_fake_per_quartile} per quartile × {NUM_QUARTILES} quartiles")
    
    pool_size = len(fake_images) + len(real_images) - SEED_SIZE_TOTAL - VALIDATION_SIZE_TOTAL
    print(f"\n   POOL ({pool_size:,} images):")
    print(f"      Remaining images after seed and validation")
    
    # ══════════════════════════════════════════════════════════
    # STEP 3: SETUP OUTPUT DIRECTORIES
    # ══════════════════════════════════════════════════════════
    
    print(f"\n📁 STEP 3: Setting up output directories...")
    setup_output_directories(OUTPUT_DIR)
    
    # ══════════════════════════════════════════════════════════
    # STEP 4: SPLIT FAKE IMAGES (STRATIFIED BY QUARTILE)
    # ══════════════════════════════════════════════════════════
    
    print(f"\n🎭 STEP 4: Processing FAKE images (stratified across quartiles)...")
    
    seed_fakes = []
    val_fakes = []
    pool_fakes = []
    
    quartile_stats = []
    
    for q_idx, (start, end, q_name) in enumerate(FAKE_QUARTILES):
        print(f"\n   {q_name}")
        
        # Get images from this quartile
        quartile_images = fake_images[start:end]
        print(f"      Total in quartile: {len(quartile_images):,}")
        
        if len(quartile_images) == 0:
            print(f"      ⚠️  WARNING: No images in this quartile!")
            continue
        
        # Shuffle within quartile (important!)
        random.shuffle(quartile_images)
        
        # Calculate how many to take from this quartile
        # Handle last quartile which might have remainder
        q_seed_count = seed_fake_per_quartile
        q_val_count = val_fake_per_quartile
        
        # Add remainder to last quartile
        if q_idx == NUM_QUARTILES - 1:
            q_seed_count += seed_fake_count % NUM_QUARTILES
            q_val_count += val_fake_count % NUM_QUARTILES
        
        # Split this quartile
        q_seed = quartile_images[:q_seed_count]
        q_val = quartile_images[q_seed_count:q_seed_count + q_val_count]
        q_pool = quartile_images[q_seed_count + q_val_count:]
        
        print(f"      → Seed: {len(q_seed)}")
        print(f"      → Validation: {len(q_val)}")
        print(f"      → Pool: {len(q_pool):,}")
        
        seed_fakes.extend(q_seed)
        val_fakes.extend(q_val)
        pool_fakes.extend(q_pool)
        
        # Save stats
        quartile_stats.append({
            'quartile': q_idx + 1,
            'name': q_name,
            'range': f"{start}-{end}",
            'total': len(quartile_images),
            'seed': len(q_seed),
            'validation': len(q_val),
            'pool': len(q_pool)
        })
    
    print(f"\n   ✅ Total FAKE distribution:")
    print(f"      Seed: {len(seed_fakes):,}")
    print(f"      Validation: {len(val_fakes):,}")
    print(f"      Pool: {len(pool_fakes):,}")
    print(f"      Total: {len(seed_fakes) + len(val_fakes) + len(pool_fakes):,}")
    
    # ══════════════════════════════════════════════════════════
    # STEP 5: SPLIT REAL IMAGES (SIMPLE RANDOM)
    # ══════════════════════════════════════════════════════════
    
    print(f"\n👤 STEP 5: Processing REAL images (random split)...")
    
    # Shuffle real images
    random.shuffle(real_images)
    
    # Split
    seed_reals = real_images[:seed_real_count]
    val_reals = real_images[seed_real_count:seed_real_count + val_real_count]
    pool_reals = real_images[seed_real_count + val_real_count:]
    
    print(f"   Seed: {len(seed_reals):,}")
    print(f"   Validation: {len(val_reals):,}")
    print(f"   Pool: {len(pool_reals):,}")
    print(f"   Total: {len(seed_reals) + len(val_reals) + len(pool_reals):,}")
    
    # ══════════════════════════════════════════════════════════
    # STEP 6: COPY FILES TO OUTPUT DIRECTORIES
    # ══════════════════════════════════════════════════════════
    
    print(f"\n📂 STEP 6: Copying files to output directories...")
    print(f"   (This may take a few minutes...)\n")
    
    def copy_files(file_list, dest_dir, desc):
        """Copy files with progress bar"""
        dest_dir = Path(dest_dir)
        for f in tqdm(file_list, desc=desc, unit='img'):
            shutil.copy2(f, dest_dir / f.name)
    
    # Copy seed
    copy_files(seed_reals, Path(OUTPUT_DIR) / 'seed' / 'real', 
               "   Copying SEED (real)")
    copy_files(seed_fakes, Path(OUTPUT_DIR) / 'seed' / 'fake', 
               "   Copying SEED (fake)")
    
    # Copy validation
    copy_files(val_reals, Path(OUTPUT_DIR) / 'validation' / 'real', 
               "   Copying VALIDATION (real)")
    copy_files(val_fakes, Path(OUTPUT_DIR) / 'validation' / 'fake', 
               "   Copying VALIDATION (fake)")
    
    # Copy pool
    copy_files(pool_reals, Path(OUTPUT_DIR) / 'pool' / 'real', 
               "   Copying POOL (real)")
    copy_files(pool_fakes, Path(OUTPUT_DIR) / 'pool' / 'fake', 
               "   Copying POOL (fake)")
    
    # ══════════════════════════════════════════════════════════
    # STEP 7: VERIFY RESULTS
    # ══════════════════════════════════════════════════════════
    
    print(f"\n🔍 STEP 7: Verifying split...")
    
    verification = {}
    
    for split in ['seed', 'validation', 'pool']:
        real_count = len(list((Path(OUTPUT_DIR) / split / 'real').glob('*.*')))
        fake_count = len(list((Path(OUTPUT_DIR) / split / 'fake').glob('*.*')))
        total = real_count + fake_count
        
        verification[split] = {
            'real': real_count,
            'fake': fake_count,
            'total': total,
            'balance': f"{fake_count/total*100:.1f}% fake" if total > 0 else "N/A"
        }
    
    print(f"\n📊 Final Dataset Distribution:\n")
    for split, stats in verification.items():
        print(f"   {split.upper()}:")
        print(f"      Real: {stats['real']:>6,}")
        print(f"      Fake: {stats['fake']:>6,}")
        print(f"      ─────────────")
        print(f"      Total: {stats['total']:>5,}")
        print(f"      Balance: {stats['balance']}")
        print()
    
    # Verify totals
    total_real = sum(v['real'] for v in verification.values())
    total_fake = sum(v['fake'] for v in verification.values())
    grand_total = total_real + total_fake
    
    print(f"   GRAND TOTAL:")
    print(f"      Real: {total_real:>6,}")
    print(f"      Fake: {total_fake:>6,}")
    print(f"      ─────────────")
    print(f"      Total: {grand_total:>5,}")
    
    # Check for missing images
    expected_total = len(real_images) + len(fake_images)
    if grand_total != expected_total:
        print(f"\n   ⚠️  WARNING: Expected {expected_total:,} total images, got {grand_total:,}")
        print(f"      Missing: {expected_total - grand_total:,} images")
    else:
        print(f"\n   ✅ All images accounted for!")
    
    # ══════════════════════════════════════════════════════════
    # STEP 8: VERIFY STRATIFICATION IN SEED
    # ══════════════════════════════════════════════════════════
    
    print(f"\n🔬 STEP 8: Verifying stratification in SEED...")
    
    seed_fake_dir = Path(OUTPUT_DIR) / 'seed' / 'fake'
    seed_fake_files = set(f.name for f in seed_fake_dir.glob('*.*'))
    
    print(f"\n   Quartile representation in SEED:")
    for q_idx, (start, end, q_name) in enumerate(FAKE_QUARTILES):
        # Count how many seed images came from this quartile
        quartile_in_seed = sum(
            1 for img in fake_images[start:end] 
            if img.name in seed_fake_files
        )
        print(f"      {q_name}: {quartile_in_seed} images")
    
    # ══════════════════════════════════════════════════════════
    # STEP 9: VERIFY STRATIFICATION IN VALIDATION
    # ══════════════════════════════════════════════════════════
    
    print(f"\n🔬 STEP 9: Verifying stratification in VALIDATION...")
    
    val_fake_dir = Path(OUTPUT_DIR) / 'validation' / 'fake'
    val_fake_files = set(f.name for f in val_fake_dir.glob('*.*'))
    
    print(f"\n   Quartile representation in VALIDATION:")
    for q_idx, (start, end, q_name) in enumerate(FAKE_QUARTILES):
        quartile_in_val = sum(
            1 for img in fake_images[start:end] 
            if img.name in val_fake_files
        )
        print(f"      {q_name}: {quartile_in_val} images")
    
    # ══════════════════════════════════════════════════════════
    # STEP 10: SAVE METADATA
    # ══════════════════════════════════════════════════════════
    
    print(f"\n💾 STEP 10: Saving metadata...")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'configuration': {
            'seed_size': SEED_SIZE_TOTAL,
            'validation_size': VALIDATION_SIZE_TOTAL,
            'num_quartiles': NUM_QUARTILES,
        },
        'input': {
            'fake_images': len(fake_images),
            'real_images': len(real_images),
            'total': len(fake_images) + len(real_images)
        },
        'quartiles': quartile_stats,
        'output': verification,
        'paths': {
            'fake_dir': str(FAKE_DIR),
            'real_dir': str(REAL_DIR),
            'output_dir': str(OUTPUT_DIR)
        }
    }
    
    save_split_metadata(OUTPUT_DIR, metadata)
    
    # ══════════════════════════════════════════════════════════
    # DONE!
    # ══════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("✅ DATASET SPLIT COMPLETE!")
    print("=" * 70)
    print(f"\nOutput saved to: {OUTPUT_DIR}/")
    print(f"   ├── seed/")
    print(f"   │   ├── real/ ({verification['seed']['real']:,} images)")
    print(f"   │   └── fake/ ({verification['seed']['fake']:,} images)")
    print(f"   ├── validation/")
    print(f"   │   ├── real/ ({verification['validation']['real']:,} images)")
    print(f"   │   └── fake/ ({verification['validation']['fake']:,} images)")
    print(f"   ├── pool/")
    print(f"   │   ├── real/ ({verification['pool']['real']:,} images)")
    print(f"   │   └── fake/ ({verification['pool']['fake']:,} images)")
    print(f"   └── split_metadata.json")
    
    print(f"\n📝 Next steps:")
    print(f"   1. Verify splits: ls -la {OUTPUT_DIR}/*/")
    print(f"   2. Review metadata: cat {OUTPUT_DIR}/split_metadata.json")
    print(f"   3. Start training: python models/train_cnn.py")
    print()

# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        stratified_split_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("   Partial files may exist in output directory")
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your configuration and try again.")