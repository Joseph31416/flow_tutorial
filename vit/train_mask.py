"""
Training script for UNet model to reconstruct original images from masked MNIST images.
Uses the ViTUnet model from unet.py and MaskedMNISTDataset from masked_dataset.py.
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

from vit_unet import ViTUnet, ViTUnetResNorm
from unet import UNetNoViT
from masked_dataset import MaskedMNISTDataset
from torchvision import transforms

class MaskReconstructionTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        num_patches: int = 8,
        patch_size: int = 5,
        mask_value: float = 0.0
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mask_value = mask_value
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and split masked MNIST dataset into train, validation, and test sets."""
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Create full training dataset
        full_dataset = MaskedMNISTDataset(
            root='../data', 
            train=True, 
            transform=transform, 
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            mask_value=self.mask_value
        )
        
        # Create test dataset
        test_dataset = MaskedMNISTDataset(
            root='../data', 
            train=False, 
            transform=transform, 
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            mask_value=self.mask_value
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
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for original, masked, _ in progress_bar:
            original = original.to(self.device)
            masked = masked.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(masked)
            loss = self.criterion(reconstructed, original)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for original, masked, _ in val_loader:
                original = original.to(self.device)
                masked = masked.to(self.device)
                
                reconstructed = self.model(masked)
                loss = self.criterion(reconstructed, original)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, save_path: str = "mask_reconstruction_model.pth"):
        """Train the model for the specified number of epochs."""
        train_loader, val_loader, test_loader = self.load_data()
        
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        print(f"Device: {self.device}")
        print(f"Number of patches: {self.num_patches}")
        print(f"Patch size: {self.patch_size}x{self.patch_size}")
        print(f"Mask value: {self.mask_value}")
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
                    'num_patches': self.num_patches,
                    'patch_size': self.patch_size,
                    'mask_value': self.mask_value
                }, save_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Final test evaluation
        test_loss = self.validate(test_loader)
        print(f"\nFinal Test Loss: {test_loss:.4f}")
        
        return test_loader
    
    def plot_training_history(self, save_path: str = "mask_training_history.png"):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Mask Reconstruction: Training and Validation Loss History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training history plot saved to {save_path}")
    
    def visualize_results(self, test_loader: DataLoader, num_samples: int = 8, save_path: str = "mask_reconstruction_samples.png"):
        """Visualize reconstruction results on test samples."""
        self.model.eval()
        
        # Get a batch of test samples
        original_batch, masked_batch, labels_batch = next(iter(test_loader))
        original_batch = original_batch.to(self.device)
        masked_batch = masked_batch.to(self.device)
        
        with torch.no_grad():
            reconstructed_batch = self.model(masked_batch)
        
        # Move to CPU for plotting
        original_batch = original_batch.cpu()
        masked_batch = masked_batch.cpu()
        reconstructed_batch = reconstructed_batch.cpu()
        
        # Select random samples
        indices = np.random.choice(len(original_batch), min(num_samples, len(original_batch)), replace=False)
        
        # Create subplot grid
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
        if num_samples == 1:
            axes = axes.reshape(3, 1)
        
        for i, idx in enumerate(indices):
            # Original image
            axes[0, i].imshow(original_batch[idx].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original\nLabel: {labels_batch[idx].item()}')
            axes[0, i].axis('off')
            
            # Masked image
            axes[1, i].imshow(masked_batch[idx].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Masked\n({self.num_patches} patches)')
            axes[1, i].axis('off')
            
            # Reconstructed image
            axes[2, i].imshow(reconstructed_batch[idx].squeeze(), cmap='gray')
            
            # Calculate reconstruction error
            mse = torch.mean((original_batch[idx] - reconstructed_batch[idx]) ** 2).item()
            axes[2, i].set_title(f'Reconstructed\nMSE: {mse:.4f}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Reconstruction samples saved to {save_path}")
    
    def visualize_mask_locations(self, test_loader: DataLoader, num_samples: int = 5, save_path: str = "mask_locations.png"):
        """Visualize different random mask patterns on the same image."""
        # Get one sample image
        original_batch, _, labels_batch = next(iter(test_loader))
        sample_image = original_batch[0]  # Use first image
        sample_label = labels_batch[0].item()
        
        # Generate multiple masked versions
        fig, axes = plt.subplots(1, num_samples + 1, figsize=(2*(num_samples + 1), 2))
        
        # Show original
        axes[0].imshow(sample_image.squeeze(), cmap='gray')
        axes[0].set_title(f'Original\nLabel: {sample_label}')
        axes[0].axis('off')
        
        # Show different masked versions
        for i in range(num_samples):
            # Create a new masked version by applying random masks
            masked_image = sample_image.clone()
            C, H, W = masked_image.shape
            
            # Generate random masks
            for _ in range(self.num_patches):
                max_top = max(0, H - self.patch_size)
                max_left = max(0, W - self.patch_size)
                top = np.random.randint(0, max_top + 1)
                left = np.random.randint(0, max_left + 1)
                masked_image[:, top:top+self.patch_size, left:left+self.patch_size] = self.mask_value
            
            axes[i + 1].imshow(masked_image.squeeze(), cmap='gray')
            axes[i + 1].set_title(f'Mask Pattern {i+1}')
            axes[i + 1].axis('off')
        
        plt.suptitle(f'Random Mask Patterns: {self.num_patches} patches of {self.patch_size}x{self.patch_size}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Mask location visualization saved to {save_path}")
    
    def calculate_metrics(self, test_loader: DataLoader) -> dict:
        """Calculate various metrics on the test set."""
        self.model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for original, masked, _ in test_loader:
                original = original.to(self.device)
                masked = masked.to(self.device)
                
                reconstructed = self.model(masked)
                
                # Calculate metrics
                mse = torch.mean((original - reconstructed) ** 2)
                mae = torch.mean(torch.abs(original - reconstructed))
                
                batch_size = original.size(0)
                total_mse += mse.item() * batch_size
                total_mae += mae.item() * batch_size
                total_samples += batch_size
        
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "mps"
    print(f"Using device: {device}")
    
    # Model parameters
    in_channels = 1  # MNIST is grayscale
    out_channels = 1
    num_heads = 4
    patch_size = 7  # 28x28 images, so 7x7 patches give us 4x4 grid
    
    # Training parameters
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 10
    
    # Masking parameters
    num_patches = 8  # Number of patches to mask (K)
    mask_patch_size = 5  # Size of each masked patch (N x N)
    mask_value = 0.0  # Black masks
    
    # Initialize model
    model = ViTUnetResNorm(
        in_channels=in_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        patch_size=patch_size
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = MaskReconstructionTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_patches=num_patches,
        patch_size=mask_patch_size,
        mask_value=mask_value
    )
    
    # Train the model
    print("Starting training...")
    test_loader = trainer.train(num_epochs=num_epochs, save_path="mask_reconstruction_model.pth")
    
    # Plot training history
    trainer.plot_training_history("mask_training_history.png")
    
    # Visualize results
    trainer.visualize_results(test_loader, num_samples=8, save_path="mask_reconstruction_samples.png")
    
    # Visualize mask patterns
    trainer.visualize_mask_locations(test_loader, num_samples=5, save_path="mask_locations.png")
    
    # Calculate and print metrics
    metrics = trainer.calculate_metrics(test_loader)
    print("\nFinal Test Metrics:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
