import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate = 0.0001  # Reduced learning rate
num_epochs = 15  # Reduced epochs for faster testing
latent_dim = 128

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=128):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(data)
            # Flatten original data to match reconstructed output
            original_flat = data.view(data.size(0), -1)
            loss = criterion(reconstructed, original_flat)
            
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
                original_flat = data.view(data.size(0), -1)
                loss = criterion(reconstructed, original_flat)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
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
        
        # Reshape for visualization
        train_original = train_samples.cpu().view(-1, 28, 28)
        train_recon = train_reconstructed.cpu().view(-1, 28, 28)
        val_original = val_samples.cpu().view(-1, 28, 28)
        val_recon = val_reconstructed.cpu().view(-1, 28, 28)
    
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
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Main function to run the autoencoder training and evaluation."""
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_data()
    
    print("Initializing autoencoder...")
    model = Autoencoder(input_dim=784, latent_dim=latent_dim).to(device)
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    print(f"\nTraining autoencoder for {num_epochs} epochs...")
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

if __name__ == "__main__":
    main()