import torch

from src.losses.ReconstructionLoss import GaussianLoss, LaplaceLoss, BernoulliLoss
from src.losses.loss import kl_divergence
from src.models.factory import create_model
from pathlib import Path
import wandb
from src.utils.constants import WANDB_ENTITY, WANDB_PROJECT
from src.datasets.utils import get_train_val_test_datatloader


def train_epoch(model, loader, optimizer, device, reconstruction_loss_fn, beta=1):
    model.train()
    reconstruction_loss_fn.train()
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0


    for images in loader:
        images = images.to(device)

        optimizer.zero_grad()
        mu, logvar, x_hat = model(images)

        reconstruction_loss = reconstruction_loss_fn(x_hat, images)

        kl_loss = kl_divergence(logvar, mu)
        vae_loss = reconstruction_loss + (beta * kl_loss)

        vae_loss.backward()
        optimizer.step()

        total_loss += vae_loss.item() * images.size(0)
        total_reconstruction_loss += reconstruction_loss.item() * images.size(0)
        total_kl_loss += kl_loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples, total_reconstruction_loss / total_samples, total_kl_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, device, reconstruction_loss_fn, beta=1):
    model.eval()
    reconstruction_loss_fn.eval()
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    for images in loader:
        images = images.to(device)

        mu, logvar, x_hat = model(images)

        reconstruction_loss = reconstruction_loss_fn(x_hat, images)

        kl_loss = kl_divergence(logvar, mu)
        vae_loss = reconstruction_loss + (beta * kl_loss)

        total_loss += vae_loss.item() * images.size(0)
        total_reconstruction_loss += reconstruction_loss.item() * images.size(0)
        total_kl_loss += kl_loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples, total_reconstruction_loss / total_samples, total_kl_loss / total_samples

def train_model(model_type="mlp", seed=42, batch_size=64, image_size=64, latent_dim=64, epochs=10, learning_rate=0.001, data_dir=None, groups_json_path=None, checkpoint_path=None, beta=1, kl_annealing_epoch=0, reconstruction_loss_fn=None):
    input_shape = (3, image_size, image_size)
    torch.manual_seed(seed)

    project_root = Path(__file__).resolve().parent.parent
    if data_dir is None:
        data_dir = project_root / "data" / "cat"
    if checkpoint_path is None:
        checkpoint_path = project_root / "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_datatloader(seed, image_size, batch_size, groups_json_path, data_dir)

    #model
    vae = create_model(model_type, input_shape, latent_dim)
    vae = vae.to(device)

    if reconstruction_loss_fn is None:
        reconstruction_loss_fn = GaussianLoss()

    reconstruction_loss_fn = reconstruction_loss_fn.to(device)

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

    reconstruction_name = reconstruction_loss_fn.__class__.__name__

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
            "seed": seed,
            "batch_size": batch_size,
            "image_size": image_size,
            "latent_dim": latent_dim,
            "learning_rate": learning_rate,
            "beta": beta,
            "architecture": model_type,
            "dataset": "AFHQv2_Cats",
            "epochs": epochs,
            "reconstruction_loss": reconstruction_name,
            "kl-annealing": kl_annealing_epoch,
            "b": getattr(reconstruction_loss_fn, "b", None)
        },
    )

    best_loss = float("inf")
    best_epoch = 0
    min_reconstruction = float("inf")
    min_reconstruction_epoch = None
    min_kl = float("inf")
    min_kl_epoch = None
    checkpoint_path = checkpoint_path /  f"{model_type}_{reconstruction_name}_epoch{epochs}_latent{latent_dim}_lr{learning_rate}_beta{beta}_{run.id}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    current_beta = beta

    for epoch in range(epochs):

        if kl_annealing_epoch > 0:
            current_beta = beta * min(1.0,(epoch + 1) / kl_annealing_epoch)

        train_loss, train_reconstruction_loss, train_kl_loss = train_epoch(vae, train_dataloader, optimizer, device, reconstruction_loss_fn, current_beta)
        val_loss, val_reconstruction_loss, val_kl_loss = evaluate(vae, val_dataloader, device, reconstruction_loss_fn, current_beta)

        train_losses["loss"].append(train_loss)
        train_losses["reconstruction_loss"].append(train_reconstruction_loss)
        train_losses["kl_loss"].append(train_kl_loss)
        val_losses["loss"].append(val_loss)
        val_losses["reconstruction_loss"].append(val_reconstruction_loss)
        val_losses["kl_loss"].append(val_kl_loss)

        if val_loss < best_loss:
            torch.save({
                "epoch": epoch + 1,
                "model_type": model_type,
                "input_shape": input_shape,
                "latent_dim": latent_dim,
                "learning_rate": learning_rate,
                "beta": beta,
                "current_beta": current_beta,
                "kl_annealing_epoch": kl_annealing_epoch,

                "reconstruction_loss": reconstruction_name,
                "reconstruction_b": getattr(reconstruction_loss_fn, "b", None),

                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, checkpoint_path / "best_checkpoint.pth")
            best_loss = val_loss
            best_epoch = epoch + 1

        if val_reconstruction_loss < min_reconstruction:
            min_reconstruction = val_reconstruction_loss
            min_reconstruction_epoch = epoch + 1

        if val_kl_loss < min_kl:
            min_kl = val_kl_loss
            min_kl_epoch = epoch + 1

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_reconstruction_loss": train_reconstruction_loss,
            "train_kl_loss": train_kl_loss,
            "val_loss": val_loss,
            "val_reconstruction_loss": val_reconstruction_loss,
            "val_kl_loss": val_kl_loss,
        })

    torch.save({
        "epoch": epoch + 1,
        "model_type": model_type,
        "input_shape": input_shape,
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
        "beta": beta,
        "current_beta": current_beta,
        "kl_annealing_epoch": kl_annealing_epoch,

        "reconstruction_loss": reconstruction_name,
        "reconstruction_b": getattr(reconstruction_loss_fn, "b", None),

        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_losses["loss"][-1]
    }, checkpoint_path / "last_checkpoint.pth")

    run.summary["min_val_reconstruction_loss"] = min_reconstruction
    run.summary["min_reconstruction_epoch"] = min_reconstruction_epoch
    run.summary["min_val_kl_loss"] = min_kl
    run.summary["min_kl_epoch"] = min_kl_epoch
    run.summary["best_val_loss"] = best_loss
    run.summary["best_epoch"] = best_epoch
    run.finish()

    return vae, train_losses, val_losses

def main():
    model_type = "cnn_vector"
    beta = 1
    seed = 42
    batch_size = 64
    image_size = 64
    latent_dim = 64
    epochs = 1
    learning_rate = 0.001
    kl_annealing_epoch = 15
    reconstruction_loss_fn = GaussianLoss()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "cat"
    checkpoint_path = root / "checkpoints"
    groups_json_path = root / "src" / "datasets" / "groups_cats.json"

    vae, train_losses, val_losses = train_model(model_type, seed, batch_size, image_size, latent_dim, epochs, learning_rate, data_dir, groups_json_path, checkpoint_path, beta=beta, kl_annealing_epoch=kl_annealing_epoch, reconstruction_loss_fn=reconstruction_loss_fn)




if __name__ == "__main__":
    main()