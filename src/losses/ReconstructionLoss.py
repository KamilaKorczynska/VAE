from torch import nn
import torch
from torchmetrics.functional.image import structural_similarity_index_measure


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

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

    @property
    def name(self):
        return f"LaplaceLoss_b{self.b}"

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


class SSIMLoss(ReconstructionLoss):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x):
        similarity = structural_similarity_index_measure(x_hat, x, data_range=1.0)

        return 1 - similarity


class CombinedReconstructionLoss(ReconstructionLoss):
    def __init__(self, base_loss, additional_losses=None):
        super().__init__()
        self.base_loss = base_loss

        if additional_losses is None:
            additional_losses = []

        self.losses = nn.ModuleList([loss_fn for loss_fn, weight in additional_losses])
        self.weights = [weight for loss_fn, weight in additional_losses]

    @property
    def name(self):
        name = self.base_loss.name

        for loss_fn, weight in zip(self.losses, self.weights):
            name += f"_{loss_fn.name}_w{weight}"

        return name

    def forward(self, x_hat, x):
        loss = self.base_loss(x_hat, x)

        for loss_fn, weight in zip(self.losses, self.weights):
            additional_loss = weight * loss_fn(x_hat, x)

            # print(
            #     "base:",
            #     loss.item(),
            #     "additional:",
            #     additional_loss.item()
            # )

            loss = loss + additional_loss

        return loss