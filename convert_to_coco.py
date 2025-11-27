import json
import os

# Load the data
with open('output/data.json', 'r') as f:
    data = json.load(f)

# Categories
categories = [
    {"id": 0, "name": "rect"},
    {"id": 1, "name": "circle"},
    {"id": 2, "name": "ellipsis"}
]

# Initialize lists
images = []
annotations = []
annotation_id = 0

# Process each image
for image_id, item in enumerate(data):
    # Image info
    images.append({
        "id": image_id,
        "file_name": f"image_{image_id:04d}.png",
        "width": item["width"],
        "height": item["height"]
    })
    
    # Annotations
    for shape in item["shapes"]:
        # Determine category
        if shape["type"] == "rect":
            category_id = 0
        elif shape["type"] == "circle":
            category_id = 1
        elif shape["type"] == "ellipsis":
            category_id = 2
        else:
            continue  # Skip unknown types
        
        # Bbox: [x_min, y_min, width, height]
        x, y, w, h = shape["x"], shape["y"], shape["w"], shape["h"]
        bbox = [x - w/2, y - h/2, w, h]
        
        # Area
        area = w * h
        
        # Annotation
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
        annotation_id += 1

# COCO format
coco_data = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Save to file
with open('output/coco_annotations.json', 'w') as f:
    json.dump(coco_data, f, indent=4)

print("COCO annotations saved to output/coco_annotations.json")