import torch
from src.utils.checkpoints import load_checkpoint
import matplotlib.pyplot as plt
from pathlib import Path
from src.datasets.Cat_dataset import CatDataset, transform
from torch.utils.data import DataLoader, random_split
from src.utils.Dinov2.split_dataset import load_and_split_groups


@torch.no_grad()
def muss(model, val_loader):
    model.eval()
    mus = []

    for images in val_loader:
        mu, logvar = model.encoder(images)
        mus.append(mu)

    mus = torch.cat(mus, dim=0)

    return mus

seed = 42
batch_size = 64
image_size = 64

root = Path(__file__).resolve().parent.parent.parent
data_dir = root / "data" / "cat"
groups_json_path = root / "src" / "datasets" / "groups_cats.json"

model_beta2, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent2_lr0.001_beta2_aygun9um" / "best_checkpoint.pth")
model_beta1, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent2_lr0.001_beta1_9dr690dt" / "best_checkpoint.pth")
model_beta05, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent2_lr0.001_beta0.5_vltxhy3x" / "best_checkpoint.pth")
model_beta01, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent2_lr0.001_beta0.1_nsazfyjp" / "best_checkpoint.pth")
model_beta0, _, _, _, _ = load_checkpoint(root / "checkpoints" / "cnn_vector_mse_epoch50_latent2_lr0.001_beta0_f1qd9dev" / "best_checkpoint.pth")


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

mu_2 = muss(model_beta2, val_dataloader)
mu_1 = muss(model_beta1, val_dataloader)
mu_05 = muss(model_beta05, val_dataloader)
mu_01 = muss(model_beta01, val_dataloader)
mu_0 = muss(model_beta0, val_dataloader)


fg, ax = plt.subplots(1,5, figsize=(16,16))
ax[0].scatter(mu_2[:, 0], mu_2[:, 1], alpha=0.7)
ax[0].set_title("beta = 2")
ax[1].scatter(mu_1[:, 0], mu_1[:, 1], alpha=0.7)
ax[1].set_title("beta = 1")
ax[2].scatter(mu_05[:, 0], mu_05[:, 1], alpha=0.7)
ax[2].set_title("beta = 0.5")
ax[3].scatter(mu_01[:, 0], mu_01[:, 1], alpha=0.7)
ax[3].set_title("beta = 0.1")
ax[4].scatter(mu_0[:, 0], mu_0[:, 1], alpha=0.7)
ax[4].set_title("beta = 0")

for a in ax:
    a.set_xlim(-8, 8)
    a.set_ylim(-8, 8)

fg.tight_layout()
fg.savefig("latent_visualization.png")
plt.show()
