"""
Create a blurred MNIST dataset for training a model.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.datasets import MNIST
from torchvision.transforms import functional as F
from typing import Tuple

class BlurredMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, blur_radius: float = 2.0):
        """
        Args:
            root (str): Root directory of the dataset.
            train (bool): If True, creates dataset from training set, otherwise from test set.
            transform (callable, optional): Optional transform to be applied on a sample.
            blur_radius (float): Radius for Gaussian blur.
        """
        self.mnist_dataset = MNIST(root=root, train=train, download=True)
        self.transform = transform
        self.blur_radius = blur_radius

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.mnist_dataset[idx]
        if self.transform:
            image = self.transform(image)
        
        # Apply Gaussian blur
        blurred_image = F.gaussian_blur(image.unsqueeze(0), kernel_size=(5, 5), sigma=self.blur_radius).squeeze(0)
        
        return image, blurred_image, label

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    blurred_mnist_dataset = BlurredMNISTDataset(
        root='./data', train=True, transform=transform, blur_radius=2.0
    )
    
    # Get a sample
    original_image, blurred_image, label = blurred_mnist_dataset[0]
    print(f"Image shape: {original_image.shape}, Label: {label}")
    
    # Visualize the image
    import matplotlib.pyplot as plt
    
    # Create subplot for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot original image
    ax1.imshow(original_image.squeeze(), cmap='gray')
    ax1.set_title(f'Original - Label: {label}')
    ax1.axis('off')
    
    # Plot blurred image
    ax2.imshow(blurred_image.squeeze(), cmap='gray')
    ax2.set_title(f'Blurred (σ={blurred_mnist_dataset.blur_radius}) - Label: {label}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
