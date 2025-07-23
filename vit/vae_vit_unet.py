import torch
import torch.nn as nn
from modules import DownBlockWithResAttn, UpBlockWithResAttn
        
class VAEViTUnetResNorm(nn.Module):

    def __init__(self,
                #  in_channels: int, out_channels: int,
                 channels: list[int],
                 num_heads: list[int],
                 patch_sizes: list[int],
                 latent_dim: int,
                 init_h: int, init_w: int):
        super(VAEViTUnetResNorm, self).__init__()
        self.down_blocks = nn.ModuleList()
        in_channels = channels[0]
        for out_channels, num_head, patch_size in zip(channels[1:], num_heads, patch_sizes):
            self.down_blocks.append(
                DownBlockWithResAttn(in_channels, out_channels, num_head, patch_size, init_h, init_w)
            )
            in_channels = out_channels
            init_h //= 2
            init_w //= 2

        self.decode_w = init_w
        self.decode_h = init_h
        self.decode_channels = in_channels
        
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(in_channels * init_h * init_w, latent_dim)
        self.fc_logvar = nn.Linear(in_channels * init_h * init_w, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, in_channels * init_h * init_w)
        self.sigmoid = nn.Sigmoid()

        self.up_blocks = nn.ModuleList()
        for out_channels, num_head, patch_size in zip(
            reversed(channels[:-1]), reversed(num_heads[:-1]), reversed(patch_sizes[:-1])
        ):
            self.up_blocks.append(
                UpBlockWithResAttn(in_channels, out_channels, num_head, patch_size, init_h, init_w)
            )
            in_channels = out_channels
            init_h *= 2
            init_w *= 2
        self.up_blocks.append(
            UpBlockWithResAttn(in_channels, channels[0], num_heads[0], patch_sizes[0], init_h, init_w)
        )

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
        for down_block in self.down_blocks:
            x = down_block(x)
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # clip logvar between -2 and 2 to prevent numerical issues
        logvar = torch.clamp(logvar, -2.0, 2.0)
        # Return mean and log variance
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent vector back to the original image space.
        """
        x = self.fc_decode(z)
        x = x.view(x.size(0), self.decode_channels, self.decode_h, self.decode_w)
        for up_block in self.up_blocks:
            x = up_block(x)
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
