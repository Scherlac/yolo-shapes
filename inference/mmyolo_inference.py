
import pathlib
from ultralytics import YOLO

# Image data
IMAGE_DIR = pathlib.Path(__file__).parent.parent / "output"/"data"/"images"
LABEL_FILE = pathlib.Path(__file__).parent.parent / "output"/"data.json"

# generated model path
MODEL_PATH = pathlib.Path(__file__).parent.parent / "runs"/"obb"/"yolo_obb_shapes_training2"/"weights"/"best.pt"
MODEL_PATH = pathlib.Path(__file__).parent.parent / "best.pt"

def load_model(model_path):
    """Load a YOLO model from a file."""
    model = YOLO(model_path)
    return model

if __name__ == "__main__":
    # Load the trained YOLO OBB model
    model = load_model(MODEL_PATH)

    # Perform inference on a sample image
    sample_image_path = IMAGE_DIR / "train" / "image_0002.png"
    results = model.predict(source=str(sample_image_path), save=True)
    results[0].save(filename="annotated.png")

    # Print results
    # for result in results:
    #    print(result.obb)  # Print bounding box results
