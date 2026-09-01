import os
import shutil
import random
from pathlib import Path

def get_images(folder):
    valid_exts = {'.jpg', '.jpeg', '.png'}
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in valid_exts:
                images.append(os.path.join(root, file))
    return images

def merge_data():
    source_fresh = Path(r"D:\Meat analysis\Fresh")
    source_spoiled = Path(r"D:\Meat analysis\Spoiled")
    
    target_dir = Path("data/dataset_freshness")
    
    fresh_imgs = get_images(source_fresh)
    spoiled_imgs = get_images(source_spoiled)
    
    print(f"Found {len(fresh_imgs)} fresh images and {len(spoiled_imgs)} spoiled images in new dataset.")
    
    # Shuffle for random train/val split
    random.seed(42)
    random.shuffle(fresh_imgs)
    random.shuffle(spoiled_imgs)
    
    def split_and_copy(imgs, class_name, prefix):
        split_idx = int(len(imgs) * 0.8)
        train_imgs = imgs[:split_idx]
        val_imgs = imgs[split_idx:]
        
        for i, img_path in enumerate(train_imgs):
            dest = target_dir / "train" / class_name / f"{prefix}_new_{i}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest)
            
        for i, img_path in enumerate(val_imgs):
            dest = target_dir / "val" / class_name / f"{prefix}_new_{i}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest)
            
        print(f"Copied {len(train_imgs)} {class_name} to train, {len(val_imgs)} to val.")

    split_and_copy(fresh_imgs, "good", "FRESH")
    split_and_copy(spoiled_imgs, "spoiled", "SPOILED")
    
    print("\nDataset merge complete!")

if __name__ == "__main__":
    merge_data()
