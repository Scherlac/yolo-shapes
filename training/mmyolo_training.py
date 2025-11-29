import os
import pathlib
from ultralytics import YOLO

# Image data
IMAGE_DIR = pathlib.Path(__file__).parent.parent / "output"/"data"/"png"

if __name__ == "__main__":
    # Load a pretrained YOLO model
    model = YOLO('yolo11n-obb.pt')  # Use YOLO11 OBB model

    # runs/obb/yolo_obb_shapes_training/weights/best.pt
    model_path = pathlib.Path(__file__).parent.parent / "runs"/"obb"/"yolo_obb_shapes_training"/"weights"/"best.pt"
    if model_path.exists():
        model.load(model_path) 


    # Train the model
    data_path = pathlib.Path(__file__).parent.parent / "data.yaml"
    results = model.train(
        data=str(data_path),
        task='obb',
        epochs=180,  # Reduced for testing
        imgsz=640,
        batch=12,  # Reduced batch size for CPU
        name='yolo_obb_shapes_training'
    )

    print("Training completed. Model saved in runs/obb/yolo_obb_shapes_training/")

    # Optional: Load and test on a sample image
    sample_image_path = IMAGE_DIR / "image_0000.png"  # Assuming images are named image_0000.png etc.
    if sample_image_path.exists():
        results = model(sample_image_path)
        print("Inference results on sample image:")
        for result in results:
            print("OBB detections:")
            print(result.obb.xywhr)  # center-x, center-y, width, height, angle (radians)


