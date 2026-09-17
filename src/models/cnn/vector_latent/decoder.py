import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()

        channels = output_shape[0]
        height = output_shape[1] //2 //2 //2
        weight = output_shape[2] //2 //2 //2

        flatten_dim = 128 * height * weight

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flatten_dim),
            nn.Unflatten(1, (128, height, weight)),

            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2),

            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2),

            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, padding=1, stride=2),

            nn.Sigmoid()
        )

    def forward(self, z):
        x_hat = self.decoder(z)

        return x_hat
