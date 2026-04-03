"""Inference and evaluation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import OxfordIIITPetDataset


def load_model(device):
    """Load trained multitask model."""
    model = MultiTaskPerceptionModel().to(device)
    model.eval()
    return model


def predict(model, image, device):
    """Run inference on a single image."""

    image = image.unsqueeze(0).to(device)  # [1, C, H, W]

    with torch.no_grad():
        outputs = model(image)

    # Extract outputs
    cls_logits = outputs["classification"]
    bbox = outputs["localization"]
    seg_logits = outputs["segmentation"]

    # Convert predictions
    pred_class = torch.argmax(cls_logits, dim=1).item()
    bbox = bbox.squeeze(0).cpu().numpy()
    seg_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

    return pred_class, bbox, seg_mask


def visualize(image, bbox, seg_mask):
    """Visualize results."""

    image = image.permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axs[0].imshow(image)
    axs[0].set_title("Image")

    # Bounding box
    axs[1].imshow(image)
    x_c, y_c, w, h = bbox
    x1 = x_c - w / 2
    y1 = y_c - h / 2

    rect = plt.Rectangle((x1, y1), w, h, edgecolor='r', facecolor='none', linewidth=2)
    axs[1].add_patch(rect)
    axs[1].set_title("Bounding Box")

    # Segmentation
    axs[2].imshow(seg_mask, cmap="gray")
    axs[2].set_title("Segmentation")

    for ax in axs:
        ax.axis("off")

    plt.show()


def run_inference(data_dir="data", index=0):
    """Run full pipeline on one sample."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = OxfordIIITPetDataset(root_dir=data_dir)

    # Get sample
    image, label, bbox_gt, mask_gt = dataset[index]

    # Load model
    model = load_model(device)

    # Predict
    pred_class, pred_bbox, pred_mask = predict(model, image, device)

    print(f"Predicted Class: {pred_class}")
    print(f"Predicted BBox: {pred_bbox}")

    # Visualize
    visualize(image, pred_bbox, pred_mask)


if __name__ == "__main__":
    run_inference()