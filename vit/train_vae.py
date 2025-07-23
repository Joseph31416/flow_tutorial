"""
Training script for UNet model to reconstruct original images from MNIST images.
Uses the from vit_unet import VAEViTUnetResNorm
 model from unet.py and MNISTDataset from normal_dataset.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, List
from tqdm import tqdm

from vae_vit_unet import VAEViTUnetResNorm
from normal_dataset import MNISTDataset
from torchvision import transforms

def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence loss.
    
    Args:
        mu (torch.Tensor): Mean tensor.
        logvar (torch.Tensor): Log variance tensor.
        
    Returns:
        torch.Tensor: KL divergence loss.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def recon_loss(reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """
    Compute the reconstruction loss (MSE).
    
    Args:
        reconstructed (torch.Tensor): Reconstructed image tensor.
        original (torch.Tensor): Original image tensor.
        
    Returns:
        torch.Tensor: Reconstruction loss.
    """
    # return nn.functional.binary_cross_entropy(reconstructed, original, reduction='mean')
    return nn.functional.mse_loss(reconstructed, original, reduction='mean')

class VAETrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        beta: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.beta = beta

        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and split MNIST dataset into train, validation, and test sets."""
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Create full training dataset
        full_dataset = MNISTDataset(
            root='../data', 
            train=True, 
            transform=transform
        )
        
        # Create test dataset
        test_dataset = MNISTDataset(
            root='../data', 
            train=False, 
            transform=transform
        )
        
        # Split training data into train and validation (80-20 split)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def criterion(self, reconstructed: torch.Tensor, original: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss: reconstruction loss + beta * KL divergence.
        Args:
            reconstructed (torch.Tensor): Reconstructed image tensor.
            original (torch.Tensor): Original image tensor.
            mu (torch.Tensor): Mean tensor from the VAE.
            logvar (torch.Tensor): Log variance tensor from the VAE.
        Returns:
            torch.Tensor: Total loss.
        """ 
        n_batch = original.size(0)
        recon_loss_value = recon_loss(reconstructed, original)
        kl_loss_value = kl_loss(mu, logvar) / n_batch  # Normalize by batch size
        return recon_loss_value + self.beta * kl_loss_value, recon_loss_value, kl_loss_value
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for original, _ in progress_bar:
            original = original.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self.model(original)
            # print min and max of reconstructed tensor
            # print(f"Reconstructed min: {x_recon.min().item()}, max: {x_recon.max().item()}")
            # Compute loss
            loss, recon_loss_value, kl_loss_value = self.criterion(x_recon, original, mu, logvar)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss_value.item()
            total_kl_loss += kl_loss_value.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        print(f"Recon Loss: {total_recon_loss / num_batches:.4f}, KL Loss: {total_kl_loss / num_batches:.4f}")
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for original, _ in val_loader:
                original = original.to(self.device)
                # Forward pass
                x_recon, mu, logvar = self.model(original)
                loss, recon_loss_value, kl_loss_value = self.criterion(x_recon, original, mu, logvar)    
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, save_path: str = "vae_reconstruction_model.pth"):
        """Train the model for the specified number of epochs."""
        train_loader, val_loader, test_loader = self.load_data()
        
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'beta': self.beta
                }, save_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Final test evaluation
        test_loss = self.validate(test_loader)
        print(f"\nFinal Test Loss: {test_loss:.4f}")
        
        return test_loader
    
    def plot_training_history(self, save_path: str = "training_history.png"):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training history plot saved to {save_path}")
    
    def visualize_results(self, test_loader: DataLoader, num_samples: int = 8, save_path: str = "vae_reconstruction_samples.png"):
        """Visualize VAE reconstruction and sampling results on test samples."""
        self.model.eval()
        
        # Get a batch of test samples
        original_batch, labels_batch = next(iter(test_loader))
        original_batch = original_batch.to(self.device)
        
        with torch.no_grad():
            # Get reconstructions from VAE forward pass
            reconstructed_batch, mu, logvar = self.model(original_batch)
            # Get samples using the sample method
            sampled_batch = self.model.sample(original_batch)
        
        # Move to CPU for plotting
        original_batch = original_batch.cpu()
        reconstructed_batch = reconstructed_batch.cpu()
        sampled_batch = sampled_batch.cpu()
        
        # Select random samples
        indices = np.random.choice(len(original_batch), min(num_samples, len(original_batch)), replace=False)
        
        # Create subplot grid: Original, Reconstructed, Sampled
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
        if num_samples == 1:
            axes = axes.reshape(3, 1)
        
        for i, idx in enumerate(indices):
            # Original image
            axes[0, i].imshow(original_batch[idx].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original\nLabel: {labels_batch[idx].item()}')
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed_batch[idx].squeeze(), cmap='gray')
            
            # Calculate reconstruction error
            mse = torch.mean((original_batch[idx] - reconstructed_batch[idx]) ** 2).item()
            axes[1, i].set_title(f'Reconstructed\nMSE: {mse:.4f}')
            axes[1, i].axis('off')
            
            # Sampled image
            axes[2, i].imshow(sampled_batch[idx].squeeze(), cmap='gray')
            
            # Calculate sampling error
            sample_mse = torch.mean((original_batch[idx] - sampled_batch[idx]) ** 2).item()
            axes[2, i].set_title(f'Sampled\nMSE: {sample_mse:.4f}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"VAE reconstruction and sampling results saved to {save_path}")
    
    def visualize_latent_interpolation(self, test_loader: DataLoader, num_steps: int = 8, save_path: str = "latent_interpolation.png"):
        """Visualize interpolation between two images in latent space."""
        self.model.eval()
        
        # Get two different images
        original_batch, labels_batch = next(iter(test_loader))
        img1 = original_batch[0:1].to(self.device)  # First image
        img2 = original_batch[1:2].to(self.device)  # Second image
        
        with torch.no_grad():
            # Encode both images to get latent representations
            mu1, logvar1 = self.model.encode(img1)
            mu2, logvar2 = self.model.encode(img2)
            
            # Sample from latent distributions
            z1 = self.model.reparameterize(mu1, logvar1)
            z2 = self.model.reparameterize(mu2, logvar2)
            
            # Create interpolation steps
            interpolated_images = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                img_interp = self.model.decode(z_interp)
                interpolated_images.append(img_interp.cpu())
        
        # Plot interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
        if num_steps == 1:
            axes = [axes]
        
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img.squeeze(), cmap='gray')
            alpha = i / (num_steps - 1)
            axes[i].set_title(f'α={alpha:.2f}')
            axes[i].axis('off')
        
        plt.suptitle(f'Latent Space Interpolation\nFrom Label {labels_batch[0].item()} to Label {labels_batch[1].item()}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Latent interpolation visualization saved to {save_path}")
    
    def visualize_random_samples(self, num_samples: int = 8, save_path: str = "random_samples.png"):
        """Generate and visualize random samples from the VAE latent space."""
        self.model.eval()
        
        with torch.no_grad():
            # Sample random latent vectors
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            # Decode to generate images
            generated_images = self.model.decode(z)
        
        # Move to CPU for plotting
        generated_images = generated_images.cpu()
        
        # Plot generated samples
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
        
        plt.suptitle('Random Samples from VAE Latent Space')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Random samples visualization saved to {save_path}")
    
    def calculate_metrics(self, test_loader: DataLoader) -> dict:
        """Calculate various metrics on the test set."""
        self.model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for original, _ in test_loader:
                original = original.to(self.device)
                # Forward pass
                x_recon, mu, logvar = self.model(original)
                # Calculate MSE and MAE
                mse = torch.mean((x_recon - original) ** 2).item()
                mae = torch.mean(torch.abs(x_recon - original)).item()
                total_mse += mse * original.size(0)
                total_mae += mae * original.size(0)
                total_samples += original.size(0)
        
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        psnr = 20 * np.log10(1.0 / np.sqrt(avg_mse))  # Assuming images are in [0, 1] range
        
        metrics = {
            'MSE': avg_mse,
            'MAE': avg_mae,
            'PSNR': psnr
        }
        
        return metrics

def main():
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "mps"
    print(f"Using device: {device}")
    
    # Model parameters
    in_channels = 1  # MNIST is grayscale
    out_channels = 1
    num_heads = 4
    patch_size = 7  # 28x28 images, so 7x7 patches give us 4x4 grid
    
    # Training parameters
    learning_rate = 1e-4
    batch_size = 64
    num_epochs = 20
    
    # Initialize model
    model = VAEViTUnetResNorm(
        channels=[1, 16, 32, 64, 128],
        num_heads=[4, 4, 8, 8],
        patch_sizes=[8, 8, 4, 2],
        latent_dim=64,
        init_h=32,  # Initial height of resized MNIST images
        init_w=32  # Initial width of resized MNIST images
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = VAETrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        beta=1e-5  # KL divergence weight
    )
    
    # Train the model
    print("Starting training...")
    test_loader = trainer.train(num_epochs=num_epochs, save_path="vae_reconstruction_model.pth")
    
    # Plot training history
    trainer.plot_training_history("training_history.png")
    
    # Visualize results
    trainer.visualize_results(test_loader, num_samples=8, save_path="vae_reconstruction_samples.png")
    
    # Visualize latent interpolation
    trainer.visualize_latent_interpolation(test_loader, num_steps=8, save_path="latent_interpolation.png")
    
    # Visualize random samples from latent space
    trainer.visualize_random_samples(num_samples=8, save_path="random_samples.png")
    
    # Calculate and print metrics
    metrics = trainer.calculate_metrics(test_loader)
    print("\nFinal Test Metrics:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
