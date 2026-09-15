import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets.Cat_dataset import CatDataset, transform
from src.models.factory import create_model
from pathlib import Path
from torch.utils.data import Subset
from scripts.train import train_epoch
from src.utils.reconstructions import get_reconstructions, save_reconstructions
from src.utils.plots import loss_plot

def main():
    seed = 42
    torch.manual_seed(seed)
    model_type = "mlp"
    batch_size = 64
    input_shape = (3, 64, 64)
    latent_dim = 64
    epochs = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "Cats"

    transforms = transform(image_size=64)
    g = torch.Generator().manual_seed(seed)

    full_dataset = CatDataset(
        image_dir=DATA_DIR,
        transform=transforms
    )

    small_subset = Subset(full_dataset, range(10))

    train_dataloader = DataLoader(
        small_subset,
        batch_size=batch_size,
        shuffle=True,
        generator = g
    )

    vae = create_model(model_type, input_shape, latent_dim)
    vae = vae.to(device)

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    train_losses = []
    reconstruction_losses = []
    kl_losses = []
    for epoch in range(epochs):
        train_loss, reconstruction_loss, kl_loss = train_epoch(vae, train_dataloader, optimizer, criterion, device)

        train_losses.append(train_loss)
        reconstruction_losses.append(reconstruction_loss)
        kl_losses.append(kl_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {train_loss:.4f} | "
        )

    images, x_hats = get_reconstructions(vae, train_dataloader, device)
    save_reconstructions(images, x_hats, PROJECT_ROOT / "results" / "overfitting_test")

    loss_plot(reconstruction_losses, kl_losses, train_losses)

if __name__ == "__main__":
    main()