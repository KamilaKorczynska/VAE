import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.datasets.Cat_dataset import CatDataset, transform
from src.losses.loss import kl_divergence, calculate_reconstruction_loss
from src.models.VAE import VAE
from pathlib import Path
import wandb
from src.utils.constants import WANDB_ENTITY, WANDB_PROJECT
from src.utils.checkpoints import best_loss_last_epoch_checkpoint


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0


    for images in loader:
        images = images.to(device)

        optimizer.zero_grad()
        mu, logvar, x_hat = model(images)

        reconstruction_loss = calculate_reconstruction_loss(x_hat, images, criterion)

        kl_loss = kl_divergence(logvar, mu)
        vae_loss = reconstruction_loss + kl_loss

        vae_loss.backward()
        optimizer.step()

        total_loss += vae_loss.item() * images.size(0)
        total_reconstruction_loss += reconstruction_loss.item() * images.size(0)
        total_kl_loss += kl_loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples, total_reconstruction_loss / total_samples, total_kl_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    for images in loader:
        images = images.to(device)

        mu, logvar, x_hat = model(images)

        reconstruction_loss = calculate_reconstruction_loss(x_hat, images, criterion)

        kl_loss = kl_divergence(logvar, mu)
        vae_loss = reconstruction_loss + kl_loss

        total_loss += vae_loss.item() * images.size(0)
        total_reconstruction_loss += reconstruction_loss.item() * images.size(0)
        total_kl_loss += kl_loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples, total_reconstruction_loss / total_samples, total_kl_loss / total_samples


def main():
    seed = 42
    batch_size = 64
    input_shape = (3, 64, 64)
    latent_dim = 64
    epochs = 10
    learning_rate = 0.001

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

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    train_losses = {
        "reconstruction_loss": [],
        "kl_loss": [],
        "loss": []
    }
    val_losses = {
        "reconstruction_loss": [],
        "kl_loss": [],
        "loss": []
    }

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
            "learning_rate": learning_rate,
            "architecture": "basic",
            "dataset": "Cats",
            "epochs": epochs
        },
    )


    checkpoint_path = PROJECT_ROOT / "checkpoints"
    best_loss, last_epoch = best_loss_last_epoch_checkpoint(checkpoint_path)

    for epoch in range(epochs):
        train_loss, train_reconstruction_loss, train_kl_loss = train_epoch(vae, train_dataloader, optimizer, criterion, device)
        val_loss, val_reconstruction_loss, val_kl_loss = evaluate(vae, val_dataloader, criterion, device)

        train_losses["loss"].append(train_loss)
        train_losses["reconstruction_loss"].append(train_reconstruction_loss)
        train_losses["kl_loss"].append(train_kl_loss)
        val_losses["loss"].append(val_loss)
        val_losses["reconstruction_loss"].append(val_reconstruction_loss)
        val_losses["kl_loss"].append(val_kl_loss)

        if val_loss < best_loss:
            torch.save({
                "epoch": epoch +1,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, checkpoint_path / "best_checkpoint.pth")
            best_loss = val_loss

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        run.log({
            "epoch": epoch +1,
            "train_loss": train_loss,
            "train_reconstruction_loss": train_reconstruction_loss,
            "train_kl_loss": train_kl_loss,
            "val_loss": val_loss,
            "val_reconstruction_loss": val_reconstruction_loss,
            "val_kl_loss": val_kl_loss,
        })

    torch.save({
        "epoch": epochs,
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_losses["loss"][-1]
    }, checkpoint_path / "last_checkpoint.pth")

    run.finish()




if __name__ == "__main__":
    main()