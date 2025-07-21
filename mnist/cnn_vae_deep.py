import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder - outputs mean and log variance for Gaussian latent space
        self.encoder_features = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 32 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 4 x 4
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 256 x 2 x 2
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 512 x 1 x 1
            nn.ReLU(), 
            nn.Flatten(),  # 512 x 1 x 1 = 512
        )
        
        # Separate layers for mean and log variance
        self.fc_mu = nn.Linear(512, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(512, latent_dim)  # Log variance
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1, 1)),  # Reshape to 512 x 1 x 1
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # 256 x 2 x 2
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 128 x 4 x 4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 64 x 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 1 x 32 x 32
            nn.Sigmoid()
        )
        
        # Add a final layer to crop from 32x32 to 28x28
        self.final_crop = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0),  # 1 x 28 x 28
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent parameters (mean and log variance)."""
        features = self.encoder_features(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode to get mean and log variance
        mu, logvar = self.encode(x)
        # Sample from the latent distribution
        z = self.reparameterize(mu, logvar)
        # Decode
        decoded = self.decoder(z)
        # Final crop to match original size
        decoded = self.final_crop(decoded)
        return decoded, mu, logvar
    
    def decode(self, z):
        """Decode from latent space."""
        decoded = self.decoder(z)
        decoded = self.final_crop(decoded)
        return decoded