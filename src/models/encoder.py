import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, flatten_input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        e = self.encoder(x)

        mu = self.mu(e)
        logvar = self.logvar(e)

        return mu, logvar