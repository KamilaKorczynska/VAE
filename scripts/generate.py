import torch
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.checkpoints import load_checkpoint

@torch.no_grad()
def generate_image(vae, latent):
    vae.eval()

    return vae.decoder(latent)


def generate_16images(checkpoint_name, latent_dim=64):
    root = Path(__file__).resolve().parent.parent
    checkpoint_path1 = root / "checkpoints" / checkpoint_name / "best_checkpoint.pth"
    model, _, _, _ = load_checkpoint(checkpoint_path1)

    generator = torch.Generator().manual_seed(42)

    latents = torch.randn(16, latent_dim, generator=generator)
    generated = generate_image(model, latents)

    fg, axs = plt.subplots(4, 4, figsize=(16, 16))

    for i, axs in enumerate(axs.flat):
        image = generated[i]

        axs.imshow(image.permute(1, 2, 0))
        axs.axis("off")

    plt.tight_layout()
    plt.savefig("generated_sample_x.png", dpi=300, bbox_inches="tight")
    plt.show()