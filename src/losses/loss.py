import torch



def vae_loss(rec_loss, kl_loss):
    return rec_loss + kl_loss

def kl_divergence(logvar, mu):
    kl = 0.5 * ( torch.exp(logvar) + mu**2 - 1 - logvar)
    kl = kl.sum(dim=1)  #sum up latent space
    return kl.mean()    #mean by batch

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
    loss = loss.sum(dim=(1,2,3))
    loss = loss.mean()
    return loss