"""
Training script for UNet model to reconstruct original images from blurred MNIST images.
Uses the ViTUnet model from unet.py and BlurredMNISTDataset from blurred_dataset.py.
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

from unet import UNetNoViT
from blurred_dataset import BlurredMNISTDataset
from torchvision import transforms

class BlurReconstructionTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        blur_radius: float = 2.0
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.blur_radius = blur_radius
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and split blurred MNIST dataset into train, validation, and test sets."""
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Create full training dataset
        full_dataset = BlurredMNISTDataset(
            root='../data', 
            train=True, 
            transform=transform, 
            blur_radius=self.blur_radius
        )
        
        # Create test dataset
        test_dataset = BlurredMNISTDataset(
            root='../data', 
            train=False, 
            transform=transform, 
            blur_radius=self.blur_radius
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
        for original, blurred, _ in progress_bar:
            original = original.to(self.device)
            blurred = blurred.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(blurred)
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
            for original, blurred, _ in val_loader:
                original = original.to(self.device)
                blurred = blurred.to(self.device)
                
                reconstructed = self.model(blurred)
                loss = self.criterion(reconstructed, original)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, save_path: str = "blur_reconstruction_model.pth"):
        """Train the model for the specified number of epochs."""
        train_loader, val_loader, test_loader = self.load_data()
        
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        print(f"Device: {self.device}")
        print(f"Blur radius: {self.blur_radius}")
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
                    'blur_radius': self.blur_radius
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
    
    def visualize_results(self, test_loader: DataLoader, num_samples: int = 8, save_path: str = "reconstruction_samples.png"):
        """Visualize reconstruction results on test samples."""
        self.model.eval()
        
        # Get a batch of test samples
        original_batch, blurred_batch, labels_batch = next(iter(test_loader))
        original_batch = original_batch.to(self.device)
        blurred_batch = blurred_batch.to(self.device)
        
        with torch.no_grad():
            reconstructed_batch = self.model(blurred_batch)
        
        # Move to CPU for plotting
        original_batch = original_batch.cpu()
        blurred_batch = blurred_batch.cpu()
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
            
            # Blurred image
            axes[1, i].imshow(blurred_batch[idx].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Blurred\n(σ={self.blur_radius})')
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
    
    def calculate_metrics(self, test_loader: DataLoader) -> dict:
        """Calculate various metrics on the test set."""
        self.model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for original, blurred, _ in test_loader:
                original = original.to(self.device)
                blurred = blurred.to(self.device)
                
                reconstructed = self.model(blurred)
                
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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    num_epochs = 5
    blur_radius = 2.0
    
    # Initialize model
    model = UNetNoViT(
        in_channels=in_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        patch_size=patch_size
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = BlurReconstructionTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        blur_radius=blur_radius
    )
    
    # Train the model
    print("Starting training...")
    test_loader = trainer.train(num_epochs=num_epochs, save_path="blur_reconstruction_model.pth")
    
    # Plot training history
    trainer.plot_training_history("training_history.png")
    
    # Visualize results
    trainer.visualize_results(test_loader, num_samples=8, save_path="reconstruction_samples.png")
    
    # Calculate and print metrics
    metrics = trainer.calculate_metrics(test_loader)
    print("\nFinal Test Metrics:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
