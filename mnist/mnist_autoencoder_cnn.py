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
batch_size = 128
learning_rate = 1e-4
num_epochs = 30
latent_dim = 16

class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 32 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 4 x 4
            nn.ReLU(),
            nn.Flatten(),  # 128 * 4 * 4 = 2048
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.Sigmoid()
        )
        
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
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        # Final crop to match original size
        decoded = self.final_crop(decoded)
        return decoded
    
    def decode(self, z):
        """Decode from latent space."""
        decoded = self.decoder(z)
        decoded = self.final_crop(decoded)
        return decoded

def load_data():
    """Load and split MNIST dataset into train and validation sets."""
    # Define transforms - keep values in [0, 1] range for sigmoid output
    transform = transforms.Compose([
        transforms.ToTensor()  # Already normalizes to [0, 1] range
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

def train_autoencoder(model, train_loader, val_loader, num_epochs):
    """Train the autoencoder."""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = './mnist/models/best_cnn_autoencoder.pth'
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                reconstructed = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model weights if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} ⭐ (Best)')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Load the best model weights before returning
    print(f"\nLoading best model weights (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(best_model_path))
    
    return train_losses, val_losses

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
        train_reconstructed = model(train_samples)
        val_reconstructed = model(val_samples)
        
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

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves."""
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

def main():
    """Main function to run the CNN autoencoder training and evaluation."""
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_data()
    
    print("Initializing CNN autoencoder...")
    model = CNNAutoencoder(latent_dim=latent_dim).to(device)
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    print(f"\nTraining CNN autoencoder for {num_epochs} epochs...")
    train_losses, val_losses = train_autoencoder(model, train_loader, val_loader, num_epochs)
    
    print("\nTraining completed!")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Plot reconstruction samples
    print("Displaying reconstruction samples...")
    plot_reconstruction_samples(model, train_loader, val_loader)
    
    # Print final losses
    print(f"\nFinal Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    
    # Compare with fully connected version
    print("\n" + "="*50)
    print("CNN vs Fully Connected Autoencoder Comparison:")
    print("="*50)
    print(f"CNN Autoencoder Parameters: {count_parameters(model):,}")
    print("This CNN version should:")
    print("- Preserve spatial structure better")
    print("- Be more parameter efficient")
    print("- Handle translation invariance")
    print("- Produce sharper reconstructions")

if __name__ == "__main__":
    main()
