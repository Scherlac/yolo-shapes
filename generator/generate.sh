#!/bin/bash

cd "$(dirname "$0")"/..

# Clean previous output
rm -rf output/data
mkdir -p output/data

# Generate synthetic training data
python training_generator/generate_syn.py

# Convert to COCO format
python training_generator/convert_to_coco.py

# Convert COCO to YOLO format
python training_generator/convert_coco_to_yolo.py

# Create train/val splits (if needed)
# (This step can be added later if required)
python training_generator/split_data.py