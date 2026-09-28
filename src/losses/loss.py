import torch
from torchmetrics.functional.image import structural_similarity_index_measure

def vae_loss(rec_loss, kl_loss, beta=1):
    return rec_loss + (beta * kl_loss)

def kl_divergence(logvar, mu):
    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1 - logvar)
    kl = kl.flatten(start_dim=1)
    kl = kl.sum(dim=1)
    return kl.mean()

def kl_annealing(time):
    beta = 1 / time

    return beta