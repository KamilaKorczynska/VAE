import torch


def vae_loss(rec_loss, kl_loss, beta=1):
    return rec_loss + (beta * kl_loss)

def kl_divergence(logvar, mu):
    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1 - logvar)
    kl = kl.flatten(start_dim=1)
    kl = kl.sum(dim=1)
    return kl.mean()

def kl_anneling(time):
    beta = 1 / time

    return beta


def get_reconstruction_criterion(loss_type):
    if loss_type == "gaussian":
        return torch.nn.MSELoss(reduction="none")

    elif loss_type == "laplace":
        return torch.nn.L1Loss(reduction="none")

    elif loss_type == "bernoulli":
        return torch.nn.BCELoss(reduction="none")

    else:
        raise ValueError(f"Unknown reconstruction loss: {loss_type}")


def calculate_reconstruction_loss(x_hat, x, criterion, loss_type="gaussian", b=1.0):
    loss = criterion(x_hat, x)

    if loss_type == "gaussian":
        loss = 0.5 * loss.sum(dim=(1, 2, 3))

    elif loss_type == "laplace":
        loss = (1 / b) * loss.sum(dim=(1, 2, 3))

    elif loss_type == "bernoulli":
        loss = loss.sum(dim=(1, 2, 3))

    return loss.mean()