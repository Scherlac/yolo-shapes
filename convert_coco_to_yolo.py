import json
import os
from pathlib import Path

# Load COCO data
with open('output/coco_annotations.json', 'r') as f:
    coco = json.load(f)

# Create labels directory
labels_dir = Path('output/data/labels')
labels_dir.mkdir(parents=True, exist_ok=True)

# Group annotations by image_id
annotations_by_image = {}
for ann in coco['annotations']:
    image_id = ann['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(ann)

# Process each image
for image in coco['images']:
    image_id = image['id']
    file_name = image['file_name']
    width = image['width']
    height = image['height']
    
    # Label file path
    label_file = labels_dir / f"{Path(file_name).stem}.txt"
    
    # Write annotations
    with open(label_file, 'w') as f:
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x_min, y_min, w, h]
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                w_norm = bbox[2] / width
                h_norm = bbox[3] / height
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("YOLO labels created in output/data/labels/")

# Create data.yaml
data_yaml = f"""
train: output/data/png
val: output/data/png  # Using same for now, split if needed

nc: 3
names: ['rect', 'circle', 'ellipsis']
"""

with open('data.yaml', 'w') as f:
    f.write(data_yaml.strip())

print("data.yaml created.")