# count_real_images.py
from pathlib import Path
from collections import Counter

def count_real_images(data_dir='data/raw'):
    """
    Count the number of images in the real directory
    """
    real_dir = Path(data_dir) / 'real'
    
    if not real_dir.exists():
        print(f"❌ Directory not found: {real_dir}")
        return
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Count all image files
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(real_dir.glob(f'*{ext}')))
        all_images.extend(list(real_dir.glob(f'*{ext.upper()}')))
    
    total = len(all_images)
    
    # Count by extension
    ext_counter = Counter([img.suffix.lower() for img in all_images])
    
    print(f"📊 Image count in {real_dir}:")
    print(f"   Total images: {total}")
    
    if ext_counter:
        print(f"\n   Breakdown by extension:")
        for ext, count in sorted(ext_counter.items()):
            print(f"     {ext}: {count}")
    else:
        print("   No images found!")

if __name__ == '__main__':
    count_real_images()

