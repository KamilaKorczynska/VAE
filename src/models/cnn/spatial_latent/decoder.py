import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape, latent_height=8, latent_width=8):
        super().__init__()

        latent_channels = 16 #latent_dim // (latent_height * latent_width)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_channels, out_channels=128, kernel_size=1),

            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2),

            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2),

            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=output_shape[0], kernel_size=4, padding=1, stride=2),

            nn.Sigmoid()
        )

    def forward(self, z):
        x_hat = self.decoder(z)

        return x_hat
