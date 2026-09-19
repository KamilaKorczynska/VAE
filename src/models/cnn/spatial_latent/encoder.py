import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, latent_height=8, latent_width=8):
        super().__init__()

        in_channels = input_shape[0]

        latent_channels = 16 #latent_dim // (latent_height * latent_width)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),

        )
        self.mu = nn.Conv2d(in_channels=128, out_channels=latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(in_channels=128, out_channels=latent_channels, kernel_size=1)

    def forward(self, x):
        e = self.encoder(x)

        mu = self.mu(e)
        logvar = self.logvar(e)

        return mu, logvar