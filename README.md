# yolo-shapes

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Scherlac/yolo-shapes.git
   cd yolo-shapes
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Linux/Mac: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

This will install all necessary packages, including Ultralytics YOLOv11 for OBB training, PyTorch for model handling, and libraries for data generation and image processing.

## How to Run

### 1. Generate Synthetic Data (Optional)
If you need to generate new synthetic shape data, run the data generation script:
```bash
python training_generator/generate_syn.py
```
This will create images in `output/data/png/` and annotations in `output/data.json`.

### 2. Convert Data to OBB YOLO Format
Convert the custom data format directly to YOLO OBB format:
```bash
python training_generator/convert_json_to_obb_yolo.py
```
This generates OBB label files in `output/data/labels/` with format `class x1 y1 x2 y2 x3 y3 x4 y4`.

### 3. Prepare Train/Val Split
Split the data into training and validation sets:
```bash
python split_data.py
```
This organizes images and labels into `output/data/images/train/`, `output/data/images/val/`, `output/data/labels/train/`, and `output/data/labels/val/`, and updates `data.yaml`.

### 4. Start Training
Run the training script:
```bash
python training/mmyolo_training.py
```
This will train a YOLO11 OBB model on the shape dataset. Adjust epochs and batch size in the script as needed. Trained models are saved in `runs/detect/`.

### Notes
- Pre-generated data is already available in the repository.
- For inference, load the trained model and run predictions on new images. Use `result.obb.xywhr` for oriented bounding box outputs.
- Monitor training progress in the console output and validation metrics.
- OBB provides better accuracy for rotated shapes compared to axis-aligned bounding boxes.