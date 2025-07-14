"""
CNN Variational Autoencoder with Gaussian Regularization for MNIST

This script implements a Convolutional Variational Autoencoder (CNN VAE) that regularizes
the latent space to follow an isotropic Gaussian distribution. Key features:

1. Encoder outputs mean (μ) and log-variance (log σ²) for each latent dimension
2. Reparameterization trick for differentiable sampling: z = μ + σ * ε, where ε ~ N(0,I)
3. KL divergence loss regularizes latent space: KL(q(z|x) || p(z)) where p(z) = N(0,I)
4. Total loss: Reconstruction Loss + β * KL Divergence Loss
5. Enables generation of new samples by sampling from N(0,I) in latent space

The β parameter controls the trade-off between reconstruction quality and latent space regularity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device with preference: cuda -> mps -> cpu
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")

# Hyperparameters
batch_size = 256
learning_rate = 1e-3
num_epochs = 20
latent_dim = 32
beta = 0.01  # Weight for KL divergence loss (can be tuned)
binarization_threshold = 0.5  # Threshold for converting grayscale to binary images

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
            nn.Flatten(),  # 128 * 4 * 4 = 2048
        )
        
        # Separate layers for mean and log variance
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)  # Log variance
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # Reshape to 128 x 4 x 4
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

def load_data():
    """Load and split MNIST dataset into train and validation sets."""
    # Define transforms - binarize images for better BCE loss performance
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range
        # transforms.Lambda(lambda x: (x > binarization_threshold).float())  # Binarize: 1 if > threshold, 0 otherwise
    ])
    
    # Download and load MNIST dataset
    full_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split training data into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def kl_divergence_loss(mu, logvar):
    """
    Compute KL divergence loss between latent distribution and standard normal.
    KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_loss.mean()

def train_autoencoder(model, train_loader, val_loader, num_epochs):
    """Train the variational autoencoder with Gaussian regularization."""
    reconstruction_criterion = nn.BCELoss()  # Binary Cross Entropy for pixel values in [0, 1]
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_recon_losses = []
    val_kl_losses = []
    
    best_val_loss = float('inf')
    best_model_path = './mnist/models/best_cnn_autoencoder.pth'
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(data)
            
            # Compute losses
            recon_loss = reconstruction_criterion(reconstructed, data)
            kl_loss = kl_divergence_loss(mu, logvar)
            total_loss = recon_loss + beta * kl_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                reconstructed, mu, logvar = model(data)
                
                # Compute losses
                recon_loss = reconstruction_criterion(reconstructed, data)
                kl_loss = kl_divergence_loss(mu, logvar)
                total_loss = recon_loss + beta * kl_loss
                
                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_kl_loss)
        val_recon_losses.append(val_recon_loss)
        val_kl_losses.append(val_kl_loss)
        
        # Save best model weights if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}), Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, KL: {val_kl_loss:.6f}) ⭐ (Best)')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}), Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, KL: {val_kl_loss:.6f})')
    
    # Load the best model weights before returning
    print(f"\nLoading best model weights (Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(torch.load(best_model_path))
    
    return (train_losses, val_losses, train_recon_losses, train_kl_losses, 
            val_recon_losses, val_kl_losses)

def plot_reconstruction_samples(model, train_loader, val_loader):
    """Plot original and reconstructed images from train and validation sets."""
    model.eval()
    
    # Get samples from train and validation sets
    train_data, _ = next(iter(train_loader))
    val_data, _ = next(iter(val_loader))
    
    # Take first 2 samples from each
    train_samples = train_data[:2].to(device)
    val_samples = val_data[:2].to(device)
    
    with torch.no_grad():
        # Reconstruct samples
        train_reconstructed, _, _ = model(train_samples)
        val_reconstructed, _, _ = model(val_samples)
        
        # Move to CPU for visualization
        train_original = train_samples.cpu().squeeze()
        train_recon = train_reconstructed.cpu().squeeze()
        val_original = val_samples.cpu().squeeze()
        val_recon = val_reconstructed.cpu().squeeze()
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Plot train samples
    for i in range(2):
        # Original
        axes[0, i*2].imshow(train_original[i], cmap='gray')
        axes[0, i*2].set_title(f'Train Original {i+1}')
        axes[0, i*2].axis('off')
        
        # Reconstructed
        axes[0, i*2+1].imshow(train_recon[i], cmap='gray')
        axes[0, i*2+1].set_title(f'Train Reconstructed {i+1}')
        axes[0, i*2+1].axis('off')
    
    # Plot validation samples
    for i in range(2):
        # Original
        axes[1, i*2].imshow(val_original[i], cmap='gray')
        axes[1, i*2].set_title(f'Val Original {i+1}')
        axes[1, i*2].axis('off')
        
        # Reconstructed
        axes[1, i*2+1].imshow(val_recon[i], cmap='gray')
        axes[1, i*2+1].set_title(f'Val Reconstructed {i+1}')
        axes[1, i*2+1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, train_recon_losses=None, train_kl_losses=None, 
                         val_recon_losses=None, val_kl_losses=None):
    """Plot training and validation loss curves."""
    if train_recon_losses is not None:
        # Plot detailed loss curves with reconstruction and KL components
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('CNN VAE: Total Loss (Reconstruction + β×KL)')
        ax1.legend()
        ax1.grid(True)
        
        # Reconstruction loss
        ax2.plot(range(1, len(train_recon_losses) + 1), train_recon_losses, label='Training Recon Loss')
        ax2.plot(range(1, len(val_recon_losses) + 1), val_recon_losses, label='Validation Recon Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Reconstruction Loss')
        ax2.set_title('CNN VAE: Reconstruction Loss')
        ax2.legend()
        ax2.grid(True)
        
        # KL divergence loss
        ax3.plot(range(1, len(train_kl_losses) + 1), train_kl_losses, label='Training KL Loss')
        ax3.plot(range(1, len(val_kl_losses) + 1), val_kl_losses, label='Validation KL Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('KL Divergence Loss')
        ax3.set_title(f'CNN VAE: KL Divergence Loss (β={beta})')
        ax3.legend()
        ax3.grid(True)
        
        # Beta-weighted KL loss
        ax4.plot(range(1, len(train_kl_losses) + 1), [beta * x for x in train_kl_losses], label='Training β×KL Loss')
        ax4.plot(range(1, len(val_kl_losses) + 1), [beta * x for x in val_kl_losses], label='Validation β×KL Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('β × KL Divergence Loss')
        ax4.set_title(f'CNN VAE: Weighted KL Loss (β={beta})')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
    else:
        # Simple plot for backward compatibility
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CNN Autoencoder: Training and Validation Loss')
        plt.legend()
        plt.grid(True)
    
    plt.show()

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_latent_space_samples(model, num_samples=16):
    """Generate and plot samples from the latent space."""
    model.eval()
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Generate samples
        generated = model.decode(z)
        generated = generated.cpu().squeeze()
    
    # Plot generated samples
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(num_samples):
        row, col = i // 4, i % 4
        axes[row, col].imshow(generated[i], cmap='gray')
        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Samples from Standard Normal Latent Space')
    plt.tight_layout()
    plt.show()

def analyze_latent_space(model, test_loader, num_samples=1000):
    """Analyze the learned latent space distribution."""
    model.eval()
    
    latent_means = []
    latent_logvars = []
    
    with torch.no_grad():
        count = 0
        for data, _ in test_loader:
            if count >= num_samples:
                break
                
            data = data.to(device)
            mu, logvar = model.encode(data)
            
            latent_means.append(mu.cpu())
            latent_logvars.append(logvar.cpu())
            
            count += data.size(0)
    
    # Concatenate all samples
    all_means = torch.cat(latent_means, dim=0)[:num_samples]
    all_logvars = torch.cat(latent_logvars, dim=0)[:num_samples]
    all_stds = torch.exp(0.5 * all_logvars)
    
    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean statistics
    mean_stats = all_means.mean(dim=0)
    std_stats = all_means.std(dim=0)
    
    axes[0, 0].bar(range(latent_dim), mean_stats.numpy())
    axes[0, 0].set_title('Mean of Latent Dimensions')
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Target (0)')
    axes[0, 0].legend()
    
    axes[0, 1].bar(range(latent_dim), std_stats.numpy())
    axes[0, 1].set_title('Standard Deviation of Latent Dimensions')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Target (1)')
    axes[0, 1].legend()
    
    # Variance statistics
    var_means = torch.exp(all_logvars).mean(dim=0)
    var_stds = torch.exp(all_logvars).std(dim=0)
    
    axes[1, 0].bar(range(latent_dim), var_means.numpy())
    axes[1, 0].set_title('Mean of Predicted Variances')
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Mean Variance')
    axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Target (1)')
    axes[1, 0].legend()
    
    # Histogram of first latent dimension
    axes[1, 1].hist(all_means[:, 0].numpy(), bins=50, alpha=0.7, density=True, label='Encoded')
    x = np.linspace(-3, 3, 100)
    y = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    axes[1, 1].plot(x, y, 'r-', label='Standard Normal', linewidth=2)
    axes[1, 1].set_title('Distribution of First Latent Dimension')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nLatent Space Analysis (β={beta}):")
    print(f"Mean of means: {mean_stats.mean().item():.6f} (target: 0.0000)")
    print(f"Std of means: {std_stats.mean().item():.6f} (target: ~1.0000)")
    print(f"Mean of predicted variances: {var_means.mean().item():.6f} (target: 1.0000)")
    print(f"KL divergence regularization strength (β): {beta}")

def main():
    """Main function to run the CNN autoencoder training and evaluation."""
    print("Loading MNIST dataset...")
    print(f"Binarizing images with threshold: {binarization_threshold}")
    train_loader, val_loader, test_loader = load_data()
    
    print("Initializing CNN autoencoder...")
    model = CNNAutoencoder(latent_dim=latent_dim).to(device)
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    print(f"\nTraining CNN VAE for {num_epochs} epochs with β={beta}...")
    result = train_autoencoder(model, train_loader, val_loader, num_epochs)
    train_losses, val_losses, train_recon_losses, train_kl_losses, val_recon_losses, val_kl_losses = result
    
    print("\nTraining completed!")
    
    # Plot training curves with detailed loss breakdown
    plot_training_curves(train_losses, val_losses, train_recon_losses, train_kl_losses,
                        val_recon_losses, val_kl_losses)
    
    # Plot reconstruction samples
    print("Displaying reconstruction samples...")
    plot_reconstruction_samples(model, train_loader, val_loader)
    
    # Generate samples from latent space
    print("Generating samples from latent space...")
    plot_latent_space_samples(model, num_samples=16)
    
    # Analyze latent space distribution
    print("Analyzing latent space distribution...")
    analyze_latent_space(model, test_loader, num_samples=1000)
    
    # Print final losses
    print(f"\nFinal Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Validation Loss: {val_losses[-1]:.6f}")
    
    # Compare with standard autoencoder
    print("\n" + "="*60)
    print("CNN Variational Autoencoder with Gaussian Regularization:")
    print("="*60)
    print(f"CNN VAE Parameters: {count_parameters(model):,}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"KL Regularization Strength (β): {beta}")
    print(f"Image Binarization Threshold: {binarization_threshold}")
    print("This CNN VAE version:")
    print("- Uses binarized MNIST images (0 or 1 pixel values)")
    print("- Employs Binary Cross Entropy loss for reconstruction")
    print("- Regularizes latent space to follow isotropic Gaussian distribution")
    print("- Enables generation of new samples from the latent space")
    print("- Provides smooth interpolation in latent space")
    print("- Balances reconstruction quality with latent space regularity")
    print("- Uses reparameterization trick for differentiable sampling")

if __name__ == "__main__":
    main()
