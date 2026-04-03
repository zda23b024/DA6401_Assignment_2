"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(self, root_dir: str, split: str = "train"):
        """
        Args:
            root_dir: path to dataset folder
            split: "train" or "test"
        """
        self.root_dir = root_dir
        self.split = split

        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.masks_dir = os.path.join(self.annotations_dir, "trimaps")
        self.xml_dir = os.path.join(self.annotations_dir, "xmls")

        # Get file list
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) if f.endswith(".jpg")
        ])

        # Build label mapping
        self.breed_to_label = {}
        self._build_label_map()

        # Transform (IMPORTANT: normalize)
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _build_label_map(self):
        breeds = sorted(set([f.split("_")[0] for f in self.image_files]))
        self.breed_to_label = {breed: i for i, breed in enumerate(breeds)}

    def __len__(self):
        return len(self.image_files)

    def _load_bbox(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to [x_center, y_center, width, height]
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Paths
        img_path = os.path.join(self.images_dir, img_name)
        xml_path = os.path.join(self.xml_dir, img_name.replace(".jpg", ".xml"))
        mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Load mask (trimap)
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # Apply transform
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        # Convert mask to long (for loss)
        mask = mask.long()

        # Label (breed)
        breed = img_name.split("_")[0]
        label = self.breed_to_label[breed]
        label = torch.tensor(label, dtype=torch.long)

        # Bounding box
        bbox = self._load_bbox(xml_path)

        return image, label, bbox, mask