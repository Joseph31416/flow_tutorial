import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Hyperparameters
latent_dim = 32
batch_size = 512
lr = 1e-3
epochs = 10
input_dim = 784  # MNIST images flattened

# Load MNIST
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), 
                          batch_size=batch_size, shuffle=True)

# Encoder network
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 400)
        self.fc3 = nn.Linear(400, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_sigma = nn.Linear(128, latent_dim)
        self.fc_L_prime = nn.Linear(128, latent_dim * latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)
        L_prime = self.fc_L_prime(h).view(-1, latent_dim, latent_dim)
        return mu, log_sigma, L_prime

# Decoder network
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 400)
        self.fc4 = nn.Linear(400, 512)
        self.fc_out = nn.Linear(512, input_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc_out(h))

# Instantiate models
encoder = Encoder().to(device)
decoder = Decoder().to(device)

# Optimizer
optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

# Training loop
for epoch in range(epochs):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(-1, input_dim).to(device)
        batch_size = x.size(0)

        # Forward pass
        mu, log_sigma, L_prime = encoder(x)
        sigma = torch.exp(log_sigma)

        # Build full covariance using Cholesky factor
        M = torch.triu(torch.ones(latent_dim, latent_dim, device=device), diagonal=1)
        L = L_prime * M + torch.diag_embed(sigma)
        
        eps = torch.randn(batch_size, latent_dim, 1, device=device)
        z = (L @ eps).squeeze(-1) + mu

        # Decoder output
        px = decoder(z)

        # Log terms
        log_qz = -0.5 * torch.sum(eps.squeeze(-1)**2 + np.log(2 * np.pi) + 2 * log_sigma, dim=1)
        log_pz = -0.5 * torch.sum(z**2 + np.log(2 * np.pi), dim=1)
        log_px = -torch.sum(x * torch.log(px + 1e-9) + (1 - x) * torch.log(1 - px + 1e-9), dim=1)

        # ELBO
        loss = torch.mean(log_px + log_pz - log_qz)

        # Backward
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, ELBO: {total_loss / len(train_loader):.4f}")

# Load test set for evaluation
test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), 
                         batch_size=4, shuffle=True)

# Generate reconstructed samples
encoder.eval()
decoder.eval()
with torch.no_grad():
    # Get a batch of 4 test samples
    test_batch, _ = next(iter(test_loader))
    test_batch = test_batch.view(-1, input_dim).to(device)
    
    # Encode to get latent representation
    mu, log_sigma, L_prime = encoder(test_batch)
    sigma = torch.exp(log_sigma)
    
    # Build full covariance using Cholesky factor
    M = torch.triu(torch.ones(latent_dim, latent_dim, device=device), diagonal=1)
    L = L_prime * M + torch.diag_embed(sigma)
    
    # Sample from latent distribution
    eps = torch.randn(4, latent_dim, 1, device=device)
    z = (L @ eps).squeeze(-1) + mu
    
    # Decode to get reconstructions
    reconstructed = decoder(z)
    
    # Move to CPU for plotting
    original = test_batch.view(4, 28, 28).cpu()
    reconstructed = reconstructed.view(4, 28, 28).cpu()
    
    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        # Original images
        axes[0, i].imshow(original[i], cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('vae_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Reconstruction visualization saved as 'vae_reconstructions.png'")
