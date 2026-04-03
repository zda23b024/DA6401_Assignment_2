"""Training entrypoint"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier

# 🔹 ADD THESE IMPORTS (make sure these files exist)
from models.localization import LocalizerModel
from models.unet import UNet


def train_classifier(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train VGG11 classifier with W&B logging."""

    wandb.init(
        project="da6401_assignment2",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": "VGG11"
        }
    )

    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGG11Classifier(num_classes=37).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels, _, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

    save_path = "checkpoints/classifier.pth"
    torch.save(model.state_dict(), save_path)

    print(f"✅ Classifier saved at {save_path}")

    artifact = wandb.Artifact("classifier_model", type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

    wandb.finish()


# =========================================================
# 🔹 NEW: LOCALIZER TRAINING
# =========================================================
def train_localizer(data_dir, epochs=10, batch_size=32, lr=1e-4,
                    device="cuda" if torch.cuda.is_available() else "cpu"):

    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = LocalizerModel().to(device)

    criterion = nn.MSELoss()  # bounding box regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, _, bboxes, _ in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)

            outputs = model(images)
            loss = criterion(outputs, bboxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Localizer] Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    save_path = "checkpoints/localizer.pth"
    torch.save(model.state_dict(), save_path)

    print(f"✅ Localizer saved at {save_path}")


# =========================================================
# 🔹 NEW: U-NET TRAINING
# =========================================================
def train_unet(data_dir, epochs=10, batch_size=16, lr=1e-4,
               device="cuda" if torch.cuda.is_available() else "cpu"):

    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = UNet().to(device)

    criterion = nn.BCEWithLogitsLoss()  # segmentation mask
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, _, _, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[UNet] Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    save_path = "checkpoints/unet.pth"
    torch.save(model.state_dict(), save_path)

    print(f"✅ UNet saved at {save_path}")


# =========================================================
# 🔹 MAIN
# =========================================================
if __name__ == "__main__":
    data_dir = "data"

    train_classifier(data_dir=data_dir)
    train_localizer(data_dir=data_dir)
    train_unet(data_dir=data_dir)
