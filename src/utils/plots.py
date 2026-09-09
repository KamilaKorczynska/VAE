import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.basic import color


def plot(loss):
    x = [i for i in range(len(loss))]
    plt.plot(x, loss)
    plt.show()

    plt.plot(x, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


def loss_plot(reconstruction_loss, kl_loss, loss, epoch):
    x = [i for i in range(len(loss))]

    fig, ax = plt.subplots()

    ax.plot(x, reconstruction_loss, label="Reconstruction loss")
    ax.plot(x, kl_loss, label="KL loss")
    ax.plot(x, loss, label="VAE loss")

    ax.legend()
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.savefig("losses.png")
    plt.show()

def train_val_plot(reconstruction_loss, kl_loss, loss, epoch):
    x = [i for i in range(len(loss))]

    fig, ax = plt.subplots()

    ax.plot(x, reconstruction_loss, label="train loss")
    ax.plot(x, kl_loss, label="val loss")

    ax.legend()
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("train loss vs validation loss")

    fig.savefig("train_vs_val.png")
    plt.show()