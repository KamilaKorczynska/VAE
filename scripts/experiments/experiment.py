from pathlib import Path
import torch

from scripts.experiments.experiment_config import ExperimentConfig
from scripts.experiments.vae_experiment import VAEExperiment
from scripts.experiments.experiments_comparison import ExperimentComparison
from src.datasets.utils import val_images_RQ2
from src.losses.ReconstructionLoss import GaussianLoss

root = Path(__file__).resolve().parent.parent

config = ExperimentConfig(
    name="",
    model_type="cnn_vector",
    optimizer="Adam",
    seed=42,
    batch_size=64,
    image_size=64,
    epochs=50,
    learning_rate=0.001,
    dataset="full",
    reconstruction_loss=GaussianLoss(),

    kl_annealing_epochs=0,
    latent_dim=64,
    beta=0,
    stochastic_trains=False,
)

experiments = []

experiment = VAEExperiment(config=config, root=root)
experiment.train()
experiments.append(experiment)

test_data = val_images_RQ2()

comparison = ExperimentComparison(experiments=experiments, test_data=test_data)
comparison.plot_reconstruction(mode="linear", stochastic=False)