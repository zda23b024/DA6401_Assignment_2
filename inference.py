"""
Inferences And Evaluation
"""

import os
import torch
import numpy as np
from PIL import Image

# Allowed libraries only
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def load_image(image_path):
    """Load and preprocess image."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32) / 255.0

    # Normalize (same as training)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # HWC → CHW
    image = np.transpose(image, (2, 0, 1))

    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image


def load_models(device):
    """Load all trained models."""

    classifier = VGG11Classifier(num_classes=37).to(device)
    classifier.load_state_dict(
        torch.load("checkpoints/classifier.pth", map_location=device)
    )
    classifier.eval()

    localizer = VGG11Localizer().to(device)
    localizer.load_state_dict(
        torch.load("checkpoints/localizer.pth", map_location=device)
    )
    localizer.eval()

    segmentation = VGG11UNet(num_classes=3).to(device)  # trimap = 3 classes
    segmentation.load_state_dict(
        torch.load("checkpoints/segmentation.pth", map_location=device)
    )
    segmentation.eval()

    return classifier, localizer, segmentation


def run_inference(image_path):
    """Run full pipeline inference on a single image."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    image = load_image(image_path).to(device)

    # Load models
    classifier, localizer, segmentation = load_models(device)

    with torch.no_grad():
        # ---- Classification ----
        class_logits = classifier(image)
        class_pred = torch.argmax(class_logits, dim=1).item()

        # ---- Localization ----
        bbox_pred = localizer(image).squeeze(0).cpu().numpy()

        # ---- Segmentation ----
        seg_logits = segmentation(image)
        seg_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

    return {
        "class": class_pred,
        "bbox": bbox_pred,  # [x_center, y_center, width, height]
        "mask": seg_mask,
    }


if __name__ == "__main__":
    # Sample Image usage
    image_path = "data/images/Abyssinian_1.jpg"  # Sample image


    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = run_inference(image_path)

    print("\n===== Inference Results =====")
    print("Predicted Class:", results["class"])
    print("Predicted Bounding Box:", results["bbox"])
    print("Segmentation Mask Shape:", results["mask"].shape)
