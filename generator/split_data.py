import os
import shutil
from pathlib import Path

# Paths
base_dir = Path('output/data')
images_dir = base_dir / 'png'
labels_dir = base_dir / 'labels'
train_images_dir = base_dir / 'images' / 'train'
val_images_dir = base_dir / 'images' / 'val'
train_labels_dir = base_dir / 'labels' / 'train'
val_labels_dir = base_dir / 'labels' / 'val'

# Create directories
train_images_dir.mkdir(parents=True, exist_ok=True)
val_images_dir.mkdir(parents=True, exist_ok=True)
train_labels_dir.mkdir(parents=True, exist_ok=True)
val_labels_dir.mkdir(parents=True, exist_ok=True)

# Get list of image files
image_files = sorted([f for f in images_dir.iterdir() if f.suffix == '.png'])

# Split: 80% train, 20% val
total = len(image_files)
train_count = int(0.8 * total)
train_files = image_files[:train_count]
val_files = image_files[train_count:]

# Move images and labels
for img_file in train_files:
    label_file = labels_dir / f"{img_file.stem}.txt"
    shutil.move(str(img_file), str(train_images_dir / img_file.name))
    if label_file.exists():
        shutil.move(str(label_file), str(train_labels_dir / label_file.name))

for img_file in val_files:
    label_file = labels_dir / f"{img_file.stem}.txt"
    shutil.move(str(img_file), str(val_images_dir / img_file.name))
    if label_file.exists():
        shutil.move(str(label_file), str(val_labels_dir / label_file.name))

print(f"Moved {len(train_files)} images to train, {len(val_files)} to val.")