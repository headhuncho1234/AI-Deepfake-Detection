# analyze_dataset.py
import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

def quick_visual_inspection(data_dir, num_samples=50):
    """
    Create a grid showing samples from different ranges
    to visually identify style changes
    """
    
    fake_dir = Path(data_dir) / 'fake'
    all_images = sorted(list(fake_dir.glob('*.jpg')))
    
    total = len(all_images)
    print(f"Total fake images: {total}")
    
    # Sample from different ranges
    ranges = [
        (0, 1000, "Images 0-1000"),
        (1000, 5000, "Images 1000-5000"),
        (5000, 20000, "Images 5000-20000"),
        (20000, 40000, "Images 20000-40000"),
        (40000, total, f"Images 40000-{total}")
    ]
    
    fig, axes = plt.subplots(len(ranges), 10, figsize=(20, len(ranges)*2))
    
    for row_idx, (start, end, label) in enumerate(ranges):
        # Sample 10 random images from this range
        range_images = all_images[start:end]
        samples = np.random.choice(range_images, size=min(10, len(range_images)), replace=False)
        
        for col_idx, img_path in enumerate(samples):
            img = Image.open(img_path)
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis('off')
            
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(label, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('dataset_visual_inspection.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: dataset_visual_inspection.png")
    print("   Review this image to see style changes!")

# Run it
quick_visual_inspection('data/raw')