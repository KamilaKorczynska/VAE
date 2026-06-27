# VAE

This project is my own implementation of a Variational Autoencoder (VAE), created to understand the theory behind probabilistic generative models in practice.

The goal is to implement the main VAE components manually and observe how they affect training and generation:

* encoder producing `μ` and `log σ²`
* reparameterization trick
* decoder likelihood
* reconstruction loss
* KL divergence
* ELBO-based training objective
* sampling from the prior distribution

The model is trained on the Cat Dataset from Kaggle:

https://www.kaggle.com/datasets/crawford/cat-dataset?resource=download

The dataset contains cat images, which are resized and used as unsupervised training data. After training, the decoder can be used to generate new cat-like images by sampling latent vectors from the prior distribution.

## Planned experiments

* Bernoulli likelihood with BCE loss
* Gaussian likelihood with MSE loss
* KL annealing
* different latent dimensions
* reconstruction quality comparison
* sampling from latent space
* latent interpolation
