import torch

@torch.no_grad()
def generate_image(vae, latent):
    vae.eval()

    return vae.decoder(latent)
