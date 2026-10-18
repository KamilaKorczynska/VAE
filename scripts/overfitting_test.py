import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets.Cat_dataset import CatDataset, transform
from src.losses.ReconstructionLoss import CombinedReconstructionLoss, GaussianLoss
from src.models.factory import create_model
from pathlib import Path
from torch.utils.data import Subset
from scripts.train import train_epoch, train_model
from src.utils.reconstructions import get_reconstructions, save_reconstructions, analyze_reconstructions
from src.utils.plots import loss_plot
#
# def main():
#     seed = 42
#     torch.manual_seed(seed)
#     model_type = "cnn_vector"
#     batch_size = 64
#     input_shape = (3, 64, 64)
#     latent_dim = 64
#     epochs = 500
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent
#     DATA_DIR = PROJECT_ROOT / "data" / "cat"
#
#     transforms = transform(image_size=64)
#     g = torch.Generator().manual_seed(seed)
#
#     image_paths = sorted(
#         list(Path(DATA_DIR).glob("*.jpg")) +
#         list(Path(DATA_DIR).glob("*.jpeg")) +
#         list(Path(DATA_DIR).glob("*.png"))
#     )
#
#     full_dataset = CatDataset(
#         image_paths=image_paths,
#         transform=transforms
#     )
#
#     small_subset = Subset(full_dataset, range(10))
#
#     train_dataloader = DataLoader(
#         small_subset,
#         batch_size=batch_size,
#         shuffle=True,
#         generator = g
#     )
#
#     vae = create_model(model_type, input_shape, latent_dim)
#     vae = vae.to(device)
#
#     criterion = nn.MSELoss(reduction='none')
#     optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
#
#     train_losses = []
#     reconstruction_losses = []
#     kl_losses = []
#     for epoch in range(epochs):
#         train_loss, reconstruction_loss, kl_loss = train_epoch(vae, train_dataloader, optimizer, criterion, device, beta=1)
#
#         train_losses.append(train_loss)
#         reconstruction_losses.append(reconstruction_loss)
#         kl_losses.append(kl_loss)
#
#         print(
#             f"Epoch {epoch + 1}/{epochs} | "
#             f"Train: {train_loss:.4f} | "
#         )
#
#     #images, x_hats = get_reconstructions(vae, train_dataloader, device)
#     results = analyze_reconstructions(vae, train_dataloader, device, criterion)
#     images = torch.stack([result["image"] for result in results])
#     x_hats = torch.stack([result["x_hat"] for result in results])
#     save_reconstructions(images, x_hats, PROJECT_ROOT / "results" / "CNN_VAE" / "vector_latent" / "overfitting_testttttt")
#
#     loss_plot(reconstruction_losses, kl_losses, train_losses)
#
#
# if __name__ == "__main__":
#     main()


import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.factory import create_model
from src.losses.ReconstructionLoss import GaussianLoss
from src.datasets.Cat_dataset import CatDataset, transform
from src.utils.reconstructions import save_reconstructions


# =========================
# CONFIG
# =========================

model_type = "cnn_vector"
seed = 42

batch_size = 64
image_size = 64
latent_dim = 64

epochs = 500
learning_rate = 0.001


# =========================
# PATHS
# =========================

root = Path(__file__).resolve().parent.parent

data_dir = root / "data" / "overfitting_test"

results_dir = (
    root
    / "results"
    / "CNN_VAE"
    / "vector_latent"
    / "overfitting_testttt"
)


# =========================
# SETUP
# =========================

torch.manual_seed(seed)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

input_shape = (
    3,
    image_size,
    image_size
)


# =========================
# DATASET
# =========================

transforms = transform(
    image_size=image_size
)

image_paths = sorted(
    list(data_dir.glob("*.jpg"))
    + list(data_dir.glob("*.jpeg"))
    + list(data_dir.glob("*.png"))
)

print(f"Found {len(image_paths)} images")

if len(image_paths) == 0:
    raise RuntimeError(
        f"No images found in {data_dir}"
    )

dataset = CatDataset(
    image_paths=image_paths,
    transform=transforms
)


# loader do treningu
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(seed)
)


# loader do późniejszego zapisania rekonstrukcji
# bez shuffle, żeby zachować kolejność
reconstruction_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)


# =========================
# MODEL
# =========================

vae = create_model(
    model_type=model_type,
    input_shape=input_shape,
    latent_dim=latent_dim
).to(device)

reconstruction_loss_fn = GaussianLoss().to(device)

optimizer = torch.optim.Adam(
    vae.parameters(),
    lr=learning_rate
)


# =========================
# OVERFITTING TRAINING
# =========================

for epoch in range(epochs):

    vae.train()

    total_loss = 0.0
    total_samples = 0

    for images in train_loader:

        images = images.to(device)

        optimizer.zero_grad()

        # encoder
        mu, logvar = vae.encoder(images)

        # IMPORTANT:
        # no epsilon, no sampling
        # z = mu
        x_hat = vae.decoder(mu)

        reconstruction_loss = reconstruction_loss_fn(
            x_hat,
            images
        )

        reconstruction_loss.backward()

        optimizer.step()

        batch_size_current = images.size(0)

        total_loss += (
            reconstruction_loss.item()
            * batch_size_current
        )

        total_samples += batch_size_current

    average_loss = total_loss / total_samples

    if (
        epoch == 0
        or (epoch + 1) % 10 == 0
    ):
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Reconstruction loss: {average_loss:.4f}"
        )


# =========================
# RECONSTRUCTIONS
# =========================

vae.eval()

all_images = []
all_reconstructions = []

with torch.no_grad():

    for images in reconstruction_loader:

        images = images.to(device)

        mu, logvar = vae.encoder(images)

        # again deterministic reconstruction
        x_hat = vae.decoder(mu)

        all_images.append(
            images.cpu()
        )

        all_reconstructions.append(
            x_hat.cpu()
        )


images = torch.cat(
    all_images,
    dim=0
)

x_hats = torch.cat(
    all_reconstructions,
    dim=0
)


print("Images:", images.shape)
print("Reconstructions:", x_hats.shape)


# =========================
# SAVE
# =========================

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

save_reconstructions(
    images,
    x_hats,
    results_dir
)


