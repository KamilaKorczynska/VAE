import torch
import matplotlib.pyplot as plt


class ExperimentComparison:
    def __init__(self, experiments, test_data):
        self.experiments = experiments
        self.test_data = test_data

    def plot_reconstruction(self, mode="linear", n_rows=2, n_columns=2):

        if mode == "linear":
            self._plot_reconstruction_linear()

        elif mode == "grid":
            if n_rows <= 0 or n_columns <=0 :
                raise ValueError("For grid mode n_rows and n_columns must be > 0")

            self._plot_reconstruction_grid(n_rows=n_rows, n_columns=n_columns)

        else:
            raise  ValueError(f"Unknown mode: {mode}")


    @torch.no_grad()
    def _plot_reconstruction_linear(self):
        for image_index, image in enumerate(self.test_data):
            number_of_columns = len(self.experiments) + 1

            fig, axs = plt.subplots(1, number_of_columns, figsize=(number_of_columns*3, 3))

            axs[0].imshow(image.permute(1,2,0))
            axs[0].set_title("Orginal")
            axs[0].axis("off")

            for i, experiment in enumerate(self.experiments):
                reconstruction = experiment.reconstruct(image.unsqueeze(0), stochastic=experiment.config.stochastic_reconstruction)[0]

                axs[i+1].imshow(reconstruction.permute(1, 2, 0))
                axs[i+1].set_title(experiment.config.name)
                axs[i+1].axis("off")

            fig.tight_layout()
            fig.savefig(f"reconstruction_{image_index}.png", dpi=300, bbox_inches="tight")
            plt.show()

    @torch.no_grad()
    def _plot_reconstruction_grid(self, n_rows=2, n_columns=2):
        number_of_slots = n_rows * n_columns

        if len(self.experiments) > number_of_slots:
            raise ValueError(
                f"Grid {n_rows}x{n_columns} has only {number_of_slots} slots, "
                f"but you have {len(self.experiments)} experiments."
            )

        for image_index, image in enumerate(self.test_data):

            fig = plt.figure(figsize=((n_columns + 1) * 3, n_rows * 3))
            grid = fig.add_gridspec(n_rows, n_columns + 1)

            ax_original = fig.add_subplot(grid[:, 0])
            ax_original.imshow(image.permute(1, 2, 0))
            ax_original.set_title("Original")
            ax_original.axis("off")

            for i in range(number_of_slots):
                row_index = i // n_columns
                column_index = i % n_columns

                ax = fig.add_subplot(grid[row_index, column_index + 1])

                if i >= len(self.experiments):
                    ax.axis("off")
                    continue

                experiment = self.experiments[i]
                reconstruction = experiment.reconstruct(image.unsqueeze(0), stochastic=experiment.config.stochastic_reconstruction)[0]

                ax.imshow(reconstruction.permute(1, 2, 0))
                ax.set_title(experiment.config.name)
                ax.axis("off")

            fig.tight_layout()
            fig.savefig(f"reconstruction_grid_{image_index}.png", dpi=300, bbox_inches="tight")
            plt.show()

    def _find_experiment(self, row_by, row_value, column_by, column_value):

        for experiment in self.experiments:

            if getattr(experiment.config, row_by) == row_value and getattr(experiment.config, column_by) == column_value:
                return experiment

        return None

    @torch.no_grad()
    def kl_per_dimension(self, mode="linear", n_rows=2, n_columns=2):

        if mode == "linear":
            self._kl_per_dimension_linear()

        elif mode == "grid":
            if n_rows <= 0 or n_columns <=0 :
                raise ValueError("For grid mode n_rows and n_columns must be > 0")

            self._kl_per_dimension_grid(n_rows=n_rows, n_columns=n_columns)

        else:
            raise  ValueError(f"Unknown mode: {mode}")

    def _kl_per_dimension_linear(self):
        number_of_columns = len(self.experiments)

        fig, axs = plt.subplots(1, number_of_columns, figsize=(number_of_columns*3, 3), sharey=True)

        for i, experiment in enumerate(self.experiments):
            kl = experiment.kl_per_dimension()
            axs[i].bar(range(len(kl)), kl)
            axs[i].set_title(experiment.config.name)

        fig.tight_layout()
        fig.savefig(f"kl_per_dimension.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _kl_per_dimension_grid(self, n_rows=2, n_columns=2):
        number_of_slots = n_rows * n_columns

        if len(self.experiments) > number_of_slots:
            raise ValueError(
                f"Grid {n_rows}x{n_columns} has only {number_of_slots} slots, "
                f"but you have {len(self.experiments)} experiments."
            )

        fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns * 3, n_rows * 3), sharey=True, squeeze=False)

        for i in range(number_of_slots):
            row_index = i // n_columns
            column_index = i % n_columns

            ax = axs[row_index, column_index]

            if i >= len(self.experiments):
                ax.axis("off")
                continue

            experiment = self.experiments[i]

            kl = experiment.kl_per_dimension()

            ax.bar(range(len(kl)), kl)
            ax.set_title(experiment.config.name)

        fig.tight_layout()
        fig.savefig(f"kl_per_dimension_grid_.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_losses(self, metric, mode="linear", row_by=None, column_by=None):
        if mode == "linear":
            self._plot_losses_linear(metric)

        elif mode == "grid":
            if row_by is None or column_by is None:
                raise ValueError("For grid mode row_by and column_by must be specified.")

            self._plot_losses_grid(metric, row_by=row_by, column_by=column_by)

        else:
            raise  ValueError(f"Unknown mode: {mode}")


    def _plot_losses_linear(self, metric):
        number_of_columns = len(self.experiments)

        fig, axs = plt.subplots(1, number_of_columns, figsize=(number_of_columns * 3, 3), sharey=True, sharex=True)

        for i, experiment in enumerate(self.experiments):

            if metric == "loss":
                axs[i].plot([i for i in range(1, len(experiment.history["train_loss"])+1)], experiment.history["train_loss"], label="train loss")
                axs[i].plot([i for i in range(1, len(experiment.history["val_loss"])+1)], experiment.history["val_loss"], label="validation loss")

            elif metric == "reconstruction":
                axs[i].plot([i for i in range(1, len(experiment.history["train_reconstruction_loss"])+1)], experiment.history["train_reconstruction_loss"], label="train reconstruction loss")
                axs[i].plot([i for i in range(1, len(experiment.history["val_reconstruction_loss"])+1)], experiment.history["val_reconstruction_loss"], label="validation reconstruction loss")

            elif metric == "kl":
                axs[i].plot([i for i in range(1, len(experiment.history["train_kl_loss"])+1)], experiment.history["train_kl_loss"], label="train KL loss")
                axs[i].plot([i for i in range(1, len(experiment.history["val_kl_loss"])+1)], experiment.history["val_kl_loss"], label="validation KL loss")

            else:
                raise ValueError(f"Unknown metric: {metric}")

            axs[i].set_title(experiment.config.name)
            axs[i].legend()

        fig.tight_layout()
        fig.savefig(f"plot_losses_{metric}.png", dpi=300, bbox_inches="tight")
        plt.show()


    def _plot_losses_grid(self, metric, row_by, column_by):
        row_values = sorted({getattr(experiment.config, row_by) for experiment in self.experiments})
        column_values = sorted({getattr(experiment.config, column_by) for experiment in self.experiments})

        n_rows = len(row_values)
        n_columns = len(column_values)

        fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns * 3, n_rows * 3), sharey=True, sharex=True)

        for row_index, row_value in enumerate(row_values):
            for column_index, column_value in enumerate(column_values):
                experiment = self._find_experiment(row_by, row_value, column_by, column_value)

                if experiment is None:
                    axs[row_index, column_index].axis("off")
                    continue

                if metric == "loss":
                    axs[row_index, column_index].plot([i for i in range(1, len(experiment.history["train_loss"]) + 1)], experiment.history["train_loss"], label="train loss")
                    axs[row_index, column_index].plot([i for i in range(1, len(experiment.history["val_loss"]) + 1)], experiment.history["val_loss"], label="validation loss")

                elif metric == "reconstruction":
                    axs[row_index, column_index].plot([i for i in range(1, len(experiment.history["train_reconstruction_loss"]) + 1)], experiment.history["train_reconstruction_loss"], label="train reconstruction loss")
                    axs[row_index, column_index].plot([i for i in range(1, len(experiment.history["val_reconstruction_loss"]) + 1)], experiment.history["val_reconstruction_loss"], label="validation reconstruction loss")

                elif metric == "kl":
                    axs[row_index, column_index].plot([i for i in range(1, len(experiment.history["train_kl_loss"]) + 1)], experiment.history["train_kl_loss"], label="train KL loss")
                    axs[row_index, column_index].plot([i for i in range(1, len(experiment.history["val_kl_loss"]) + 1)], experiment.history["val_kl_loss"], label="validation KL loss")

                else:
                    raise ValueError(f"Unknown metric: {metric}")
                axs[row_index, column_index].set_title(experiment.config.name)
                axs[row_index, column_index].legend()

        fig.tight_layout()
        fig.savefig(f"plot_losses_{metric}_grid_.png", dpi=300, bbox_inches="tight")
        plt.show()