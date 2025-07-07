# imports
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms

def load_data(batch_size: int = 128):
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