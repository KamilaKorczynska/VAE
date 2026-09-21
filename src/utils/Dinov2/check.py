import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def show_group(group, image_dir):
    n = len(group)

    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows)
    )

    # żeby zawsze dało się iterować
    if n == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for ax, filename in zip(axs, group):
        image = Image.open(
            Path(image_dir) / filename
        ).convert("RGB")

        ax.imshow(image)
        ax.set_title(filename)
        ax.axis("off")

    # wyłącz niewykorzystane pola
    for ax in axs[len(group):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

with open("../../datasets/groups_cats.json", "r") as file:
    groups = json.load(file)

sorted_groups = sorted(groups, key=len, reverse=True)

print("Number of groups:", len(groups))

for i, group in enumerate(sorted_groups[:10], start=1):
    print(f"Group {i}: {len(group)} images")

root = Path(__file__).resolve().parent.parent.parent.parent
image_dir = root / "data" / "cat"

for i, group in enumerate(sorted_groups[:10], start=1):
    print(f"\nGroup {i}: {len(group)} images")
    show_group(group, image_dir)