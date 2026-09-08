import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.datasets.Cat_dataset import CatDataset, transform
from src.losses.loss import kl_divergence, calculate_reconstruction_loss
from src.models.VAE import VAE
from pathlib import Path


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images in loader:
        images = images.to(device)

        optimizer.zero_grad()
        mu, logvar, ppb = model(images)

        reconstruction_loss = calculate_reconstruction_loss(ppb, images, criterion)

        kl_loss = kl_divergence(logvar, mu)
        vae_loss = reconstruction_loss + kl_loss

        vae_loss.backward()
        optimizer.step()

        total_loss += vae_loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images in loader:
        images = images.to(device)

        mu, logvar, ppb = model(images)

        reconstruction_loss = calculate_reconstruction_loss(ppb, images, criterion)

        kl_loss = kl_divergence(logvar, mu)
        vae_loss = reconstruction_loss + kl_loss

        total_loss += vae_loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples



def main():
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "Cats"

    transforms = transform(image_size=64)
    g = torch.Generator().manual_seed(seed)

    full_dataset = CatDataset(
        image_dir=DATA_DIR,
        transform=transforms
    )
    train_dataset, remaining_dataset = random_split(full_dataset, [0.7, 0.3], generator=g)
    val_dataset, test_dataset = random_split(remaining_dataset, [0.5, 0.5], generator=g)

    batch_size = 64
    input_shape = (3, 64, 64)
    latent_dim = 64
    epochs = 10

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator = g
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

    vae = VAE(input_shape, latent_dim)
    vae = vae.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_epoch(vae, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate(vae, val_dataloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
    )


if __name__ == "__main__":
    main()