import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from unet import UNet 
import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "mps"

def add_noise(images, noise_scale=0.5):
    """Add random noise to images for denoising task"""
    noise = torch.randn_like(images) * noise_scale
    noisy_images = images + noise
    # return noisy_images
    return torch.clamp(noisy_images, 0., 1.)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Create noisy input and clean target
            noisy_data = add_noise(data)
            target = data
            
            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                noisy_data = add_noise(data)
                output = model(noisy_data)
                val_loss += criterion(output, data).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def visualize_results(model, test_loader, num_samples=5):
    model.eval()
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        noisy_data = add_noise(data)
        output = model(noisy_data)
        
        # Move to CPU for visualization
        data = data.cpu()
        noisy_data = noisy_data.cpu()
        output = output.cpu()
        
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 8))
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(data[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Noisy
            axes[1, i].imshow(noisy_data[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Noisy')
            axes[1, i].axis('off')
            
            # Reconstructed
            axes[2, i].imshow(output[i].squeeze(), cmap='gray')
            axes[2, i].set_title('Denoised')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()

def filter_dataset_by_label(dataset, target_label):
    """Filter dataset to only include samples with specific label"""
    filtered_indices = []
    for idx, (_, label) in enumerate(dataset):
        if label == target_label:
            filtered_indices.append(idx)
    return torch.utils.data.Subset(dataset, filtered_indices)

def main():
    # Set device
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Filter datasets to only include label 1
    train_dataset_filtered = filter_dataset_by_label(train_dataset, target_label=3)
    test_dataset_filtered = filter_dataset_by_label(test_dataset, target_label=3)
    
    print(f"Original train dataset size: {len(train_dataset)}")
    print(f"Filtered train dataset size (label 1 only): {len(train_dataset_filtered)}")
    print(f"Original test dataset size: {len(test_dataset)}")
    print(f"Filtered test dataset size (label 1 only): {len(test_dataset_filtered)}")
    
    # Create data loaders
    test_loader = DataLoader(test_dataset_filtered, batch_size=32, shuffle=False)
    
    # Split filtered train into train/val
    train_size = int(0.8 * len(train_dataset_filtered))
    val_size = len(train_dataset_filtered) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset_filtered, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    print(f"Train loader size: {len(train_loader.dataset)}")
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    print(f"Validation loader size: {len(val_loader.dataset)}")
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1)
    print(model)
    print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train the model
    print('Starting training...')
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=2)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # Visualize results
    print('Visualizing results...')
    visualize_results(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_unet.pth')
    print('Model saved as mnist_unet.pth')

if __name__ == '__main__':
    main()
