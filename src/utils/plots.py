import matplotlib.pyplot as plt


def plot(loss):
    x = [i for i in range(len(loss))]
    plt.plot(x, loss)
    plt.show()

    plt.plot(x, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


def loss_plot(reconstruction_loss, kl_loss, loss, path=".", name="losses"):
    x = [i for i in range(len(loss))]

    fig, ax = plt.subplots()

    ax.plot(x, reconstruction_loss, label="Reconstruction loss")
    ax.plot(x, kl_loss, label="KL loss")
    ax.plot(x, loss, label="VAE loss")

    ax.legend()
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.savefig(f"{path}/{name}.png")
    plt.show()

def train_val_plot(train_loss, val_loss, path="."):
    x = [i for i in range(len(val_loss))]

    fig, ax = plt.subplots()

    ax.plot(x, train_loss, label="train loss")
    ax.plot(x, val_loss, label="val loss")

    ax.legend()
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("train loss vs validation loss")

    fig.savefig(f"{path}/train_vs_val.png")
    plt.show()