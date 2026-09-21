import torch
import json
from pathlib import Path


def split_groups(groups, train_ratio=0.7, val_ratio=0.15, seed=42):

    total_images = sum(len(group) for group in groups)

    targets = {
        "train": total_images * train_ratio,
        "val": total_images * val_ratio,
        "test": total_images * (1 - train_ratio - val_ratio)
    }

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        len(groups),
        generator=generator
    ).tolist()

    shuffled_groups = [groups[i] for i in permutation]
    shuffled_groups.sort(key=len, reverse=True)

    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }
    for group in shuffled_groups:
        deficits = {
            split_name:
                targets[split_name] - counts[split_name]

            for split_name in splits
        }

        chosen_split = max(
            deficits,
            key=deficits.get
        )

        splits[chosen_split].extend(group)
        counts[chosen_split] += len(group)

    return splits


def load_and_split_groups(groups_json_path, image_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    groups_json_path = Path(groups_json_path)
    image_dir = Path(image_dir)

    with open(groups_json_path, "r") as file:
        groups_names = json.load(file)

    groups_paths = []

    for group in groups_names:
        group_paths = []

        for filename in group:
            image_path = image_dir / filename

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            group_paths.append(image_path)

        groups_paths.append(group_paths)

    splits = split_groups(groups=groups_paths, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    return splits