"""Training entrypoint
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier


def train_classifier(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train VGG11 classifier with W&B logging."""

    # 🔹 Initialize W&B
    wandb.init(
        project="da6401_assignment2",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": "VGG11"
        }
    )

    # Dataset
    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model
    model = VGG11Classifier(num_classes=37).to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create checkpoint folder
    os.makedirs("checkpoints", exist_ok=True)

    # 🔹 Watch model (logs gradients & params)
    wandb.watch(model, log="all")

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels, _, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # 🔹 Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

    # Save model
    save_path = "checkpoints/classifier.pth"
    torch.save(model.state_dict(), save_path)

    print(f"✅ Classifier saved at {save_path}")

    # 🔹 Save model artifact in W&B
    artifact = wandb.Artifact("classifier_model", type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    train_classifier(data_dir="data")