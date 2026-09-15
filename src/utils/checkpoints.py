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


