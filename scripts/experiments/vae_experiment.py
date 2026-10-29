import torch.cuda
from torch.utils.data import DataLoader
from src.datasets.Cat_dataset import CatDataset, transform
from src.datasets.utils import get_train_val_test_datatloader, val_images_RQ2, get_dataset_RQ2
from src.losses.loss import kl_divergence
from src.models.factory import create_model
from src.utils.kl_visualization import kl_per_dimension


class VAEExperiment:
    def __init__(self, config, root, device=None):
        self.config = config
        self.root = root
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        self.reconstruction_loss = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.history = {
            "train_loss": [],
            "train_reconstruction_loss": [],
            "train_kl_loss": [],

            "val_loss": [],
            "val_reconstruction_loss": [],
            "val_kl_loss": []
        }

        self._prepare_dataset()
        self._prepare_model()

    def _prepare_dataset(self):
        if self.config.train_dataset == "full":
            data_dir = self.root / "data" / "cat"
            groups_json_path = (self.root / "src" / "datasets" / "groups_cats.json")

            self.train_loader, self.val_loader, self.test_loader = get_train_val_test_datatloader(
                                                                        self.config.seed,
                                                                        self.config.image_size,
                                                                        self.config.batch_size,
                                                                        groups_json_path,
                                                                        data_dir
                                                                        )

            if self.config.val_dataset == "RQ2":
                dataset = get_dataset_RQ2(data_dir, self.config.image_size)
                self.val_loader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )

        elif self.config.train_dataset == "overfitting":
            data_dir = (
                    self.root
                    / "data"
                    / "overfitting_test"
            )

            image_paths = sorted(
                list(data_dir.glob("*.jpg"))
                + list(data_dir.glob("*.jpeg"))
                + list(data_dir.glob("*.png"))
            )

            dataset = CatDataset(
                image_paths=image_paths,
                transform=transform(
                    image_size=self.config.image_size
                )
            )

            self.train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            self.val_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

            if self.config.val_dataset == "RQ2":
                dataset = get_dataset_RQ2()
                self.val_loader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )

    def _forward(self, images, stochastic):

        if stochastic:
            mu, logvar, x_hat = self.model(images)

        else:
            mu, logvar = self.model.encoder(images)
            x_hat = self.model.decoder(mu)

        return mu, logvar, x_hat


    def _get_beta(self, epoch):
        beta = self.config.beta

        if self.config.kl_annealing_epochs == 0:
            return beta

        return beta * min(1.0, (epoch + 1) / self.config.kl_annealing_epochs)


    def _run_epoch(self, loader, training, beta):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_rec = 0
        total_kl = 0
        total_samples = 0

        for i, images in enumerate(loader):

            images = images.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):

                if training:
                    stochastic = self.config.stochastic_train
                else:
                    stochastic = False

                mu, logvar, x_hat = self._forward(images, stochastic=stochastic)

                rec_loss = self.config.reconstruction_loss(x_hat, images)

                kl_loss = kl_divergence(logvar, mu)

                loss = rec_loss + beta * kl_loss

                if training:
                    loss.backward()
                    self.optimizer.step()

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_rec += rec_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "reconstruction_loss":
                total_rec / total_samples,
            "kl_loss":
                total_kl / total_samples
        }

    def train(self):

        print(self.config.name)
        for epoch in range(self.config.epochs):
            beta = self._get_beta(epoch)

            print(f"epoch: {epoch+1}/{self.config.epochs}")

            train_results = self._run_epoch(self.train_loader, training=True, beta=beta)
            val_results = self._run_epoch(self.val_loader, training=False, beta=beta)

            self.history["train_loss"].append(train_results["loss"])
            self.history["train_reconstruction_loss"].append(train_results["reconstruction_loss"])
            self.history["train_kl_loss"].append(train_results["kl_loss"])
            self.history["val_loss"].append(val_results["loss"])
            self.history["val_reconstruction_loss"].append(val_results["reconstruction_loss"])
            self.history["val_kl_loss"].append(val_results["kl_loss"])

    @torch.no_grad()
    def reconstruct(self, images, stochastic=False):
        self.model.eval()
        x_hat = []

        images = images.to(self.device)

        _, _, x_hat = self._forward(images, stochastic=stochastic)

        return x_hat.cpu()


    def generate(self, number_of_images=16):
        seed = self.config.seed
        self.model.eval()

        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(number_of_images,self.config.latent_dim, generator=generator).to(self.device)

        images = self.model.decoder(z)

        return images.cpu()

    def _prepare_model(self):
        image_size= self.config.image_size
        self.model = create_model(model_type=self.config.model_type,input_shape=(3, image_size, image_size), latent_dim=self.config.latent_dim).to(self.device)

        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.reconstruction_loss = self.config.reconstruction_loss.to(self.device)


    def kl_per_dimension(self):
        kl = kl_per_dimension(self.model, self.val_loader, self.device)

        return kl