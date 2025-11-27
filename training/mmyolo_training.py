import os
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


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
    # Example usage
    sample_image_path = IMAGE_DIR / "sample.png"
    image = load_image(sample_image_path)
    display_image(image, title="Sample Image")

    # Preprocess image
    image_tensor = preprocess_image(image)

    # Dummy model for demonstration
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = torch.nn.Linear(224*224*3, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = DummyModel()

    # Forward pass
    output = model(image_tensor)

    # Postprocess output
    predicted_class, probabilities = postprocess_output(output)
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Probabilities: {probabilities.detach().numpy()}")

    # Save and load model
    save_model(model, MODEL_PATH)
    loaded_model = load_model(DummyModel, MODEL_PATH)
    print("Model saved and loaded successfully.")
    return model


