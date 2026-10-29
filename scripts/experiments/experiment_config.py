from dataclasses import dataclass, field

from src.losses.ReconstructionLoss import (
    ReconstructionLoss,
    GaussianLoss
)

@dataclass
class ExperimentConfig:
    name: str

    model_type: str = "cnn_vector"
    optimizer: str = "Adam"

    seed: int = 42
    batch_size: int = 64
    image_size: int = 64
    latent_dim: int = 64

    epochs: int = 50
    learning_rate: float = 0.001

    beta: float = 1.0
    kl_annealing_epochs: int = 0

    stochastic_train: bool = True
    stochastic_reconstruction: bool = True

    train_dataset: str = "full"
    val_dataset: str = "full"

    reconstruction_loss: ReconstructionLoss = field(default_factory=GaussianLoss)