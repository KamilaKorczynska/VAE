import torch

@torch.no_grad()
def generate_image(vae, latent_dim, device):
    vae.eval()

    latent = torch.randn(1, latent_dim, device=device)
    generated = vae.decoder(latent)

    return generated
