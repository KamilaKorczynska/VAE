import torch
from src.utils.checkpoints import load_checkpoint
import matplotlib.pyplot as plt
from pathlib import Path
from src.datasets.Cat_dataset import CatDataset, transform
from torch.utils.data import DataLoader, random_split
from src.utils.Dinov2.split_dataset import load_and_split_groups


@torch.no_grad()
def kl_per_dimension(model, loader, device):
    model.eval()

    all_kl = []

    for images in loader:
        images = images.to(device)

        mu, logvar, _ = model(images)

        kl = 0.5 * (
            torch.exp(logvar)
            + mu ** 2
            - 1
            - logvar
        )

        all_kl.append(kl.cpu())

    all_kl = torch.cat(all_kl, dim=0)
    mean_kl_per_dim = all_kl.mean(dim=0)

    return mean_kl_per_dim



seed = 42
batch_size = 64
image_size = 64

root = Path(__file__).resolve().parent.parent.parent
data_dir = root / "data" / "cat"
groups_json_path = root / "src" / "datasets" / "groups_cats.json"

model_latent32, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent32_lr0.001_beta1_qeaggpm7" / "best_checkpoint.pth")
model_latent64, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent64_lr0.001_beta1_rno5o633" / "best_checkpoint.pth")
model_latent128, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent128_lr0.001_beta1_44efdlwx" / "best_checkpoint.pth")
model_latent256, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent256_lr0.001_beta1_55vl0rt3" / "best_checkpoint.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transforms = transform(image_size=image_size)
g = torch.Generator().manual_seed(seed)

splits = load_and_split_groups(
        groups_json_path=groups_json_path,
        image_dir=data_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=seed
    )

val_paths = splits["val"]

val_dataset = CatDataset(
    image_paths=val_paths,
    transform=transforms
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    generator=g
)

kl_32 = kl_per_dimension(model_latent32, val_dataloader, device)
kl_64 = kl_per_dimension(model_latent64, val_dataloader, device)
kl_128 = kl_per_dimension(model_latent128, val_dataloader, device)
kl_256 = kl_per_dimension(model_latent256, val_dataloader, device)

fg, ax = plt.subplots(1,4, figsize=(16,16))
ax[0].bar(range(len(kl_32)), kl_32)
ax[0].set_title("latent dim = 32")

ax[1].bar(range(len(kl_64)), kl_64)
ax[1].set_title("latent dim = 64")

ax[2].bar(range(len(kl_128)), kl_128)
ax[2].set_title("latent dim = 128")

ax[3].bar(range(len(kl_256)), kl_256)
ax[3].set_title("latent dim = 256")

for a in ax:
    #a.set_xlim(-8, 8)
    a.set_ylim(0, 3)

fg.tight_layout()
fg.savefig("kl_per_dimension_visualization.png")
plt.show()
