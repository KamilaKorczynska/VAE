from torch import nn
import torch


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x):
        raise NotImplementedError

class GaussianLoss(ReconstructionLoss):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction="none")

    def forward(self, x_hat, x):
        loss = self.criterion(x_hat, x)
        loss = 0.5 * loss.sum(dim=(1, 2, 3))
        return loss.mean()

class LaplaceLoss(ReconstructionLoss):
    def __init__(self, b=9):
        super().__init__()
        self.criterion = torch.nn.L1Loss(reduction="none")
        self.b = b

    def forward(self, x_hat, x):
        loss = self.criterion(x_hat, x)
        loss = (1 / self.b) * loss.sum(dim=(1, 2, 3))
        return loss.mean()

class BernoulliLoss(ReconstructionLoss):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.BCELoss(reduction="none")

    def forward(self, x_hat, x):
        loss = self.criterion(x_hat, x)
        loss = loss.sum(dim=(1, 2, 3))
        return loss.mean()