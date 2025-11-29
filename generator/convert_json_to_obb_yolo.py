import json
import os
import pathlib
import math

current_dir = pathlib.Path(__file__).parent
output_dir = current_dir.parent / "output" / "data"
shape_types=['square', 'rect', 'circle', 'ellipse']

# Load the data
with open(output_dir / 'data.json', 'r') as f:
    data = json.load(f)

# Create labels directory
labels_dir = output_dir / "labels"
labels_dir.mkdir(parents=True, exist_ok=True)

# Process each image
for i, image in enumerate(data):
    width = image['width']
    height = image['height']
    
    # Label file path
    label_file = labels_dir / f"image_{i:04d}.txt"

    
    # Write annotations
    with open(label_file, 'w') as f:
        shape : dict = None
        for shape in image["shapes"]:

            x, y, w, h = shape["x"], shape["y"], shape["w"], shape["h"]
            rot = shape.get("rot", 0)

            # Determine class index based on shape type list
            if shape['type'] in shape_types:
                cls = shape_types.index(shape['type'])
            else:
                continue
            
            
            # Compute 4 corner points before rotation (centered at origin)
            hw, hh = w / 2, h / 2
            points = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            
            # Rotate and translate
            cos_r = math.cos(rot)
            sin_r = math.sin(rot)
            rotated_points = []
            for px, py in points:
                # Rotate
                rx = px * cos_r - py * sin_r
                ry = px * sin_r + py * cos_r
                # Translate
                rx += x
                ry += y
                # Normalize
                rotated_points.extend([rx / width, ry / height])
            
            # Write line: class x1 y1 x2 y2 x3 y3 x4 y4
            line = f"{cls} {' '.join(f'{p:.6f}' for p in rotated_points)}\n"
            f.write(line)

print(f"OBB YOLO labels created in {labels_dir}")

# Create data.yaml
data_yaml = f"""
train: output/data/images/train
val: output/data/images/val

nc: 4
names: {shape_types}

task: obb
"""

with open(current_dir.parent / 'data.yaml', 'w') as f:
    f.write(data_yaml.strip())

print("data.yaml created.")