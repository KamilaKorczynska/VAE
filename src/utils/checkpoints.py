import torch
from src.models.VAE import VAE
import torch.nn as nn


def load_checkpoint(checkpoint_path, image_size, latent_dim, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_shape = (3, image_size, image_size)

    vae = VAE(
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


