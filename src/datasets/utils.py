from pathlib import Path
import shutil
import torch
from torch.utils.data import DataLoader
from src.datasets.Cat_dataset import CatDataset, transform
from src.utils.Dinov2.split_dataset import load_and_split_groups


def get_train_val_test_datatloader(seed, image_size, batch_size, groups_json_path, data_dir):

    transforms = transform(image_size=image_size)
    g = torch.Generator().manual_seed(seed)

    splits = load_and_split_groups(
        groups_json_path=groups_json_path,
        image_dir=data_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=seed
    )

    train_paths = splits["train"]
    val_paths = splits["val"]
    test_paths = splits["test"]

    train_dataset = CatDataset(
        image_paths=train_paths,
        transform=transforms
    )

    val_dataset = CatDataset(
        image_paths=val_paths,
        transform=transforms
    )

    test_dataset = CatDataset(
        image_paths=test_paths,
        transform=transforms
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=g
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=g
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":

    root = Path(__file__).resolve().parent.parent.parent
    output_dir = root / "dataset" / "val_dataset_cat"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    image_size = 64
    batch_size = 64
    groups_json_path = root / "src" / "datasets" / "groups_cats.json"
    data_dir = root / "data" / "cat"
    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_datatloader(seed, image_size, batch_size, groups_json_path, data_dir)

    for path in val_dataloader.dataset.image_paths:
        shutil.copy2(path,output_dir / path.name)


def val_images_RQ2():
    comparison_images = [
        "0206.png",
        "2014.png",
        "1973.png",
        "5297.png",
        "1903.png",
        "1980.png",
        "4132.png",
        "3304.png"
    ]

    return comparison_images