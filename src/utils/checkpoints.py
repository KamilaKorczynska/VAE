import torch

def best_loss_last_epoch_checkpoint(path):
    best_loss = float("inf")
    last_epoch = 0

    if (path / "best_checkpoint.pth").exists():
        best_checkpoint = torch.load(path / "best_checkpoint.pth")
        best_loss = best_checkpoint["val_loss"]

    if (path / "last_checkpoint.pth").exists():
        last_checkpoint = torch.load(path / "last_checkpoint.pth")
        last_epoch = last_checkpoint["epoch"]

    return best_loss, last_epoch