"""
Create a blurred MNIST dataset for training a model.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.datasets import MNIST
from torchvision.transforms import functional as F
from typing import Tuple

class MNISTDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        """
        Args:
            root (str): Root directory of the dataset.
            train (bool): If True, creates dataset from training set, otherwise from test set.
            transform (callable, optional): Optional transform to be applied on a sample.
            blur_radius (float): Radius for Gaussian blur.
        """
        self.mnist_dataset = MNIST(root=root, train=train, download=True)
        # Create a resize transform to ensure all images are 32x32
        self.resize_transform = transforms.Resize((32, 32))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.mnist_dataset[idx]
        image = self.resize_transform(image)
        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    blurred_mnist_dataset = MNISTDataset(
        root='./data', train=True, transform=transform
    )
    
    # Get a sample
    original_image, label = blurred_mnist_dataset[0]
    print(f"Image shape: {original_image.shape}, Label: {label}")
    
    # Visualize the image
    import matplotlib.pyplot as plt
    
    # Create subplot for comparison
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

    # Plot original image
    ax1.imshow(original_image.squeeze(), cmap='gray')
    ax1.set_title(f'Original - Label: {label}')
    ax1.axis('off')
    
    plt.tight_layout()
    plt.show()
