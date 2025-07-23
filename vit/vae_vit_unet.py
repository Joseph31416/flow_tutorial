import torch
import torch.nn as nn
from modules import DownBlockWithResAttn, UpBlockWithResAttn
        
class VAEViTUnetResNorm(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int, patch_size: int, latent_dim: int = 32):
        super(VAEViTUnetResNorm, self).__init__()
        self.latent_dim = latent_dim

        self.db_1 = DownBlockWithResAttn(in_channels, 64, num_heads, patch_size, 28, 28)
        self.db_2 = DownBlockWithResAttn(64, 128, num_heads, patch_size, 14, 14)
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 7 * 7)
        self.sigmoid = nn.Sigmoid()

        self.up_1 = UpBlockWithResAttn(128, 64, num_heads, patch_size, 7, 7)
        self.up_2 = UpBlockWithResAttn(64, out_channels, 7, patch_size, 14, 14)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode the input tensor into mean and log variance.
        """
        x = self.db_1(x)
        x = self.db_2(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent vector back to the original image space.
        """
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.up_1(x)
        x = self.up_2(x)
        # x = self.sigmoid(x)  # Ensure output is in the range [0, 1]
        return x
    
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from the VAE.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be of shape (B, 1, 28, 28) for MNIST-like images.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
