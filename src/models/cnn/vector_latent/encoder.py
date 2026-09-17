import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()

        in_channels = input_shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),

            nn.Flatten(),
        )

        height = input_shape[1] //2 //2 //2
        width = input_shape[2] //2 //2 //2
        flatten_shape = 128 * height * width

        self.mu = nn.Linear(flatten_shape, latent_dim)
        self.logvar = nn.Linear(flatten_shape, latent_dim)

    def forward(self, x):
        e = self.encoder(x)

        mu = self.mu(e)
        logvar = self.logvar(e)

        return mu, logvar