from src.models.mlp.MLP_VAE import MLP_VAE
from src.models.cnn.CNN_VAE import CNN_VAE

def create_model(model_type, input_shape, latent_dim):
    if model_type == "mlp":
        return MLP_VAE(input_shape=input_shape, latent_dim=latent_dim)

    if model_type == "cnn_vector":
        return CNN_VAE(input_shape=input_shape, latent_dim=latent_dim)

    raise ValueError(f"Unknown model type: {model_type}")