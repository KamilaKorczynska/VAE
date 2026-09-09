# VAE

This project is my own implementation of a Variational Autoencoder (VAE).

The goal is to build a working VAE from theory, train it, generate new images and experiment with different setups to see how they affect the output.

## Tech Stack
- **Python**
- **PyTorch**
- **Matplotlib** 


## Dataset
The model is trained on the Cat Dataset from Kaggle:
https://www.kaggle.com/datasets/crawford/cat-dataset?resource=download
 - The dataset contains cat images, which are resized (64x64 RGB) and used as unsupervised training data.

## Current Model
The current baseline VAE uses:
- an MLP-based Encoder and Decoder,
- a latent space with Gaussian posterior approximation,
- the reparameterization trick,
- Gaussian reconstruction likelihood with squared-error reconstruction loss,
- KL divergence regularization against a standard normal prior.

## Current Progress
- [x] The basic skeleton of the model is written.
- [x] **Overfitting test:** Successful. The model can overfit on a small batch (10 images) of data and the reconstructions of images were almost identical, which proves that the architecture is capable of learning.


## Planned Experiments
I want to see how different factors affect the model's performance and the quality of generated images. I plan to experiment with:
- **Reconstruction Loss:** MSE vs. BCE
- **Latent Space:** Changing its size
- **Architecture:** Modifying the number and sizes of layers in the Encoder and Decoder
- **KL Annealing:** Training with and without it
- **Normalization:** Checking if it helps the training process
