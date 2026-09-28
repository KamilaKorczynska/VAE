import torch
from src.models.factory import create_model
import torch.nn as nn


def load_checkpoint(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint["model_type"]
    input_shape = checkpoint["input_shape"]
    latent_dim = checkpoint["latent_dim"]
    learning_rate = checkpoint["learning_rate"]

    reconstruction_name = checkpoint.get("reconstruction_loss", "GaussianLoss")
    beta = checkpoint.get("beta", 1.0)
    current_beta = checkpoint.get("current_beta", beta)
    kl_annealing_epoch = checkpoint.get("kl_annealing_epoch", 0)
    reconstruction_b = checkpoint.get("reconstruction_b", 9)

    vae = create_model(
        model_type=model_type,
        input_shape=input_shape,
        latent_dim=latent_dim
    ).to(device)
    vae.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]

    training_state = {
        "beta": beta,
        "current_beta": current_beta,
        "kl_annealing_epoch": kl_annealing_epoch,

        "reconstruction_loss": reconstruction_name,
        "reconstruction_b": reconstruction_b,

        "epoch": epoch,
        "optimizer": optimizer
    }

    return vae, epoch, val_loss, training_state


def old_load_checkpoint(checkpoint_path, image_size, latent_dim, learning_rate, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_shape = (3, image_size, image_size)

    vae = create_model(
        model_type=model_type,
        input_shape=input_shape,
        latent_dim=latent_dim
    ).to(device)
    vae.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    criterion = nn.MSELoss(reduction='none')

    return vae, optimizer, criterion, epoch, val_loss


