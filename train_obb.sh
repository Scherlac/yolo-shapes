#!/bin/bash

cd "$(dirname "$0")"

# Clean up previous outputs
rm -rf output runs
rm data.json data.yaml

# Generate dataset
python training_generator/generate_syn.py

# Convert data.json to data.yaml and prepare labels for OBB
python training_generator/convert_json_to_obb_yolo.py

# Split dataset into train and val
python training_generator/split_data.py

# Train the model
python training/mmyolo_training.py