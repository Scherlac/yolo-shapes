import os
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO


# Image data
IMAGE_DIR = pathlib.Path(__file__).parent.parent / "output"/"data"/"png"
LABEL_FILE = pathlib.Path(__file__).parent.parent / "output"/"data.json"

# generated model path
MODEL_PATH = pathlib.Path(__file__).parent.parent / "output"/"models"/"best_model.pth"

def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def display_image(image, title="Image"):
    """Display an image using matplotlib."""
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_model(model, path):
    """Save a PyTorch model to a file."""
    torch.save(model.state_dict(), path)

def load_model(model_class, path):
    """Load a PyTorch model from a file."""
    model = model_class()
    model.load_state_dict(torch.load(path))

    return model

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess an image for model input."""
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_tensor = torch.tensor(image_transposed, dtype=torch.float32).unsqueeze(0)
    return image_tensor

def postprocess_output(output):
    """Postprocess model output."""
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities

if __name__ == "__main__":
    # Load a pretrained YOLO model
    model = YOLO('yolo11n-obb.pt')  # Use YOLO11 OBB model

    # Train the model
    data_path = pathlib.Path(__file__).parent.parent / "data.yaml"
    results = model.train(
        data=str(data_path),
        task='obb',
        epochs=1,  # Reduced for testing
        imgsz=640,
        batch=4,  # Reduced batch size for CPU
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


