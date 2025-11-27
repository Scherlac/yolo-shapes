# TODO List for Updating to Oriented Bounding Boxes (OBB) Training

Based on review of the project content and Ultralytics OBB documentation.

## Project Summary
- **Current Setup**: The project generates synthetic data with rotated shapes (rectangles, circles, ellipses) using `generate_syn.py`, which includes rotation (`rot`) values. However, the conversion scripts (`convert_to_coco.py` and `convert_coco_to_yolo.py`) ignore rotation and produce axis-aligned bounding boxes in standard YOLO format (class x_center y_center width height).
- **Training**: Uses Ultralytics YOLOv5 (`yolov5su.pt`) for axis-aligned object detection. The training script (`mmyolo_training.py`) is basic and loads a pretrained YOLOv5 model.
- **Data**: Shapes are stored in `data.json` with position, size, and rotation, but labels are converted to axis-aligned format.
- **Dependencies**: Includes `ultralytics`, so upgrading to OBB support is feasible.

## OBB Overview from Docs
- OBB uses rotated bounding boxes for better accuracy on angled objects (e.g., ships, vehicles in aerial imagery).
- Label format: `class_index x1 y1 x2 y2 x3 y3 x4 y4` (4 corner points, normalized 0-1).
- Internally uses `xywhr` (center x/y, width, height, rotation in radians).
- Supported in YOLOv8/YOLO11 with models like `yolo11n-obb.pt`.
- Training example: `model.train(data="dataset.yaml", epochs=100, imgsz=640)`.
- Angles are constrained to 0-90 degrees.

## Todo List
1. **Upgrade Ultralytics Version**: Update `requirements.txt` to ensure the latest Ultralytics (supports YOLO11 OBB). Run `pip install --upgrade ultralytics` to get OBB models.

2. **Modify Data Generation for OBB**: Update `generate_syn.py` to output oriented bounding boxes. Compute the 4 corner points from center (x,y), width (w), height (h), and rotation (rot). Store in `data.json` or directly generate OBB labels.

3. **Update Conversion Scripts**:
   - Modify `convert_to_coco.py` to handle rotation and compute oriented bboxes (4 points) instead of axis-aligned.
   - Update `convert_coco_to_yolo.py` to output OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4` (normalized). Skip COCO intermediate if possible.

4. **Update data.yaml**: Ensure it points to the correct paths and includes OBB-specific settings if needed (e.g., for YOLO11 OBB datasets like DOTA).

5. **Update Training Script**:
   - Change `mmyolo_training.py` to load an OBB model (e.g., `model = YOLO('yolo11n-obb.pt')`).
   - Use `model.train()` with OBB-compatible arguments (e.g., `task='obb'` if required).
   - Add validation and prediction examples using OBB outputs (e.g., `result.obb.xywhr`).

6. **Test Data Conversion**: Run the updated scripts on existing data to verify OBB labels are correct. Visualize rotated bboxes to ensure they fit the shapes.

7. **Train and Validate**: Run training with a small dataset first. Use `model.val()` to check OBB metrics (e.g., mAP for oriented boxes).

8. **Handle Edge Cases**: Ensure rotations are within 0-90 degrees as per docs. Add logic to clamp or adjust angles if needed.

9. **Documentation Update**: Update `README.md` to reflect OBB training steps, model changes, and any new dependencies.

10. **Optional Enhancements**: Integrate with DOTA dataset for pretraining if needed. Add rotation augmentation during training for better generalization.