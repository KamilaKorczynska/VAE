import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim, flatten_output_dim, input_shape):
        super().__init__()

        self.input_shape = input_shape

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, flatten_output_dim)
        )

    def forward(self, z):
        d = self.decoder(z)
        d = d.reshape(z.size(0), *self.input_shape)

        return d
