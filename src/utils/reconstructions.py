import torch
from torchvision.transforms.functional import to_pil_image

@torch.no_grad()
def get_reconstructions(model, loader, device):
    model.eval()

    images = next(iter(loader)) #one batch
    images = images.to(device)
    mu, logvar, x_hat = model(images)

    return images.cpu(), x_hat.cpu()


def show_reconstructions(images, x_hats, path="."):
    counter = 1
    for image, x_hat in zip(images, x_hats):
        img = to_pil_image(image)
        img.save(f"{path}/x{counter}.png")

        img_hat = to_pil_image(x_hat)
        img_hat.save(f"{path}/x_hat{counter}.png")

        counter += 1

