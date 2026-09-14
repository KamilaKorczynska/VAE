import torch
from torchvision.transforms.functional import to_pil_image

@torch.no_grad()
def get_reconstructions(model, loader, device):
    model.eval()

    images = next(iter(loader)) #one batch
    images = images.to(device)
    mu, logvar, x_hat = model(images)

    return images.cpu(), x_hat.cpu()


@torch.no_grad()
def analyze_reconstrustions(model, loader, device, criterion):
    model.eval()
    results = []

    for images in loader:
        images = images.to(device)

        mu, logvar = model.encoder(images)
        x_hats = model.decoder(mu)  #deterministic reconstruction: z = mu

        reconstruction_losses = criterion(x_hats, images)
        reconstruction_losses = 0.5 * reconstruction_losses.sum(dim=(1, 2, 3))

        for image, x_hat, reconstruction_loss in zip(images, x_hats, reconstruction_losses):
            results.append({
                "image": image.cpu(),
                "x_hat": x_hat.cpu(),
                "reconstruction_loss": reconstruction_loss.item()
            })

    return results


def save_reconstructions(images, x_hats, path="."):
    counter = 1
    for image, x_hat in zip(images, x_hats):
        img = to_pil_image(image)
        img.save(f"{path}/x{counter}.png")

        img_hat = to_pil_image(x_hat)
        img_hat.save(f"{path}/x_hat{counter}.png")

        counter += 1

