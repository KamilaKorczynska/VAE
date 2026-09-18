import torch.nn as nn

import torch
from src.models.cnn.spatial_latent.encoder import Encoder
from src.models.cnn.spatial_latent.decoder import Decoder

class CNN_VAE_spatial(nn.Module):
    def __init__(self, input_shape, latent_dim, latent_height=4, latent_width=4):
        super().__init__()

        self.encoder = Encoder(input_shape=input_shape, latent_dim=latent_dim, latent_height=latent_height, latent_width=latent_width)
        self.decoder = Decoder(latent_dim=latent_dim, output_shape=input_shape, latent_height=latent_height, latent_width=latent_width)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)   #std = standard devation
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        x_hat = self.decoder(z)

        return mu, logvar, x_hat
