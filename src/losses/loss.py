import torch



def vae_loss(rec_loss, kl_loss, beta=1):
    return rec_loss + (beta * kl_loss)

def kl_divergence(logvar, mu):
    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1 - logvar)
    kl = kl.flatten(start_dim=1)
    kl = kl.sum(dim=1)
    return kl.mean()

def kl_anneling():
    raise NotImplemented


def bernoulli_likelihood(pos_weight):
    criterion = torch.nn.BCEWithLogitsLoss()    #BCEWithLogitsLoss is more stable hen sigmoid + BCELoss
    return criterion

def gaussian_likelihood(pos_weight):
    criterion = torch.nn.MSELoss()
    return criterion

def calculate_reconstruction_loss(x_hat, x, criterion):
    loss = criterion(x_hat, x)
    loss = 0.5 * loss.sum(dim=(1,2,3))
    loss = loss.mean()
    return loss