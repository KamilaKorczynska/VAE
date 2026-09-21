from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted(
            list(Path(image_dir).glob("*.jpg")) +
            list(Path(image_dir).glob("*.jpeg")) +
            list(Path(image_dir).glob("*.png"))
        )

        #transformations for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, str(image_path)

def transform(image_size=64):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])