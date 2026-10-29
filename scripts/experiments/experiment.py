from pathlib import Path

from scripts.experiments.experiment_config import ExperimentConfig
from scripts.experiments.vae_experiment import VAEExperiment
from scripts.experiments.experiments_comparison import ExperimentComparison
from src.losses.ReconstructionLoss import GaussianLoss


root = Path(__file__).resolve().parent.parent.parent


epochs = 50
kl_annealing_epochs = 0

betas = [0.1, 0.5, 1]
latent_dims = [32, 64, 128, 256]


experiments = []


for beta in betas:
    for latent_dim in latent_dims:

        config = ExperimentConfig(
            name=f"beta = {beta}, latent = {latent_dim}",

            epochs=epochs,

            train_dataset="full",
            val_dataset="RQ2",

            reconstruction_loss=GaussianLoss(),

            kl_annealing_epochs=kl_annealing_epochs,

            latent_dim=latent_dim,
            beta=beta,

            stochastic_train=True,

            # nie ma znaczenia dla KL per dimension,
            # ale config może tego wymagać
            stochastic_reconstruction=False,
        )

        experiment = VAEExperiment(
            config=config,
            root=root
        )

        experiment.train()

        experiments.append(experiment)


#####################################################################

test_data = next(iter(experiments[0].val_loader))

comparison = ExperimentComparison(
    experiments=experiments,
    test_data=test_data
)

comparison.plot_reconstruction(
    mode="grid",
    n_rows=4,
    n_columns=4
)