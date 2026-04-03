"""
Training entrypoint
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localizer import LocalizerModel
from models.unet import UNet


def train_models(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train Classifier, Localizer, and UNet models with W&B logging."""

    # 🔹 Initialize W&B
    wandb.init(
        project="da6401_assignment2",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "models": ["VGG11", "Localizer", "UNet"]
        }
    )

    # Dataset
    dataset = OxfordIIITPetDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # ----------------------------
    # Classifier Model
    # ----------------------------
    classifier = VGG11Classifier(num_classes=37).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr)
    wandb.watch(classifier, log="all", log_freq=10)

    # ----------------------------
    # Localizer Model
    # ----------------------------
    localizer = LocalizerModel().to(device)
    criterion_loc = nn.MSELoss()
    optimizer_loc = torch.optim.Adam(localizer.parameters(), lr=lr)
    wandb.watch(localizer, log="all", log_freq=10)

    # ----------------------------
    # UNet Model
    # ----------------------------
    unet = UNet(in_channels=3, out_channels=1).to(device)
    criterion_unet = nn.BCEWithLogitsLoss()
    optimizer_unet = torch.optim.Adam(unet.parameters(), lr=lr)
    wandb.watch(unet, log="all", log_freq=10)

    # Create checkpoint folder
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        classifier.train()
        localizer.train()
        unet.train()

        total_loss_cls, total_loss_loc, total_loss_unet = 0.0, 0.0, 0.0

        for images, labels, bboxes, masks in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device).float()
            masks = masks.to(device).float()

            # -------- Classifier Forward + Backward --------
            optimizer_cls.zero_grad()
            outputs_cls = classifier(images)
            loss_cls = criterion_cls(outputs_cls, labels)
            loss_cls.backward()
            optimizer_cls.step()
            total_loss_cls += loss_cls.item()

            # -------- Localizer Forward + Backward --------
            optimizer_loc.zero_grad()
            outputs_loc = localizer(images)
            loss_loc = criterion_loc(outputs_loc, bboxes)
            loss_loc.backward()
            optimizer_loc.step()
            total_loss_loc += loss_loc.item()

            # -------- UNet Forward + Backward --------
            optimizer_unet.zero_grad()
            outputs_unet = unet(images)
            loss_unet = criterion_unet(outputs_unet, masks)
            loss_unet.backward()
            optimizer_unet.step()
            total_loss_unet += loss_unet.item()

        avg_loss_cls = total_loss_cls / len(dataloader)
        avg_loss_loc = total_loss_loc / len(dataloader)
        avg_loss_unet = total_loss_unet / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Classifier Loss: {avg_loss_cls:.4f}, "
            f"Localizer Loss: {avg_loss_loc:.4f}, "
            f"UNet Loss: {avg_loss_unet:.4f}"
        )

        # 🔹 Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "classifier_loss": avg_loss_cls,
            "localizer_loss": avg_loss_loc,
            "unet_loss": avg_loss_unet,
            "lr": optimizer_cls.param_groups[0]["lr"]
        })

    # Save models
    classifier_path = "checkpoints/classifier.pth"
    localizer_path = "checkpoints/localizer.pth"
    unet_path = "checkpoints/unet.pth"

    torch.save(classifier.state_dict(), classifier_path)
    torch.save(localizer.state_dict(), localizer_path)
    torch.save(unet.state_dict(), unet_path)

    print(f"✅ Classifier saved at {classifier_path}")
    print(f"✅ Localizer saved at {localizer_path}")
    print(f"✅ UNet saved at {unet_path}")

    # 🔹 Save artifacts in W&B
    artifact_cls = wandb.Artifact("classifier_model", type="model")
    artifact_cls.add_file(classifier_path)
    wandb.log_artifact(artifact_cls)

    artifact_loc = wandb.Artifact("localizer_model", type="model")
    artifact_loc.add_file(localizer_path)
    wandb.log_artifact(artifact_loc)

    artifact_unet = wandb.Artifact("unet_model", type="model")
    artifact_unet.add_file(unet_path)
    wandb.log_artifact(artifact_unet)

    wandb.finish()


if __name__ == "__main__":
    train_models(data_dir="data")
