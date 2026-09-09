import torch.nn as nn

import torch
from src.models.encoder import Encoder
from src.models.decoder import Decoder

class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()

        flatten_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.encoder = Encoder(flatten_input_dim=flatten_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, flatten_output_dim=flatten_dim, input_shape=input_shape)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)   #std = standard devation
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        x_hat = self.decoder(z)

        return mu, logvar, x_hat
