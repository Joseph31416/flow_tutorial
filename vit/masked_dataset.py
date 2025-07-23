"""
Create a masked MNIST dataset for training a model.
Randomly masks out K different N x N patches from MNIST images.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import random
from typing import Tuple

class MaskedMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, 
                 num_patches: int = 3, patch_size: int = 4, mask_value: float = 0.0):
        """
        Args:
            root (str): Root directory of the dataset.
            train (bool): If True, creates dataset from training set, otherwise from test set.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_patches (int): Number of patches (K) to mask out.
            patch_size (int): Size of each square patch (N x N).
            mask_value (float): Value to use for masked pixels (0.0 for black, 1.0 for white).
        """
        self.mnist_dataset = MNIST(root=root, train=train, download=True)
        # Create a resize transform to ensure all images are 32x32
        self.resize_transform = transforms.Resize((32, 32))
        self.transform = transform
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mask_value = mask_value

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def _apply_random_masks(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply K random N x N masks to the image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).
            
        Returns:
            torch.Tensor: Masked image tensor.
        """
        masked_image = image.clone()
        C, H, W = image.shape
        
        # Generate K random patch positions
        for _ in range(self.num_patches):
            # Ensure patch fits within image boundaries
            max_top = max(0, H - self.patch_size)
            max_left = max(0, W - self.patch_size)
            
            # Random top-left corner of the patch
            # top = random.randint(0, max_top)
            top = random.normalvariate(0.5, 0.2) * max_top  # Introduce some randomness
            top = max(0, min(int(top), max_top))  # Ensure it's within bounds
            # left = random.randint(0, max_left)
            left = random.normalvariate(0.5, 0.2) * max_left
            left = max(0, min(int(left), max_left))  # Ensure it's within bounds
            # Apply mask to the patch area
            masked_image[:, top:top+self.patch_size, left:left+self.patch_size] = self.mask_value
            
        return masked_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image, label = self.mnist_dataset[idx]
        
        # First resize the image to 32x32
        image = self.resize_transform(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Apply random masks
        masked_image = self._apply_random_masks(image)
        
        return image, masked_image, label

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    masked_mnist_dataset = MaskedMNISTDataset(
        root='./data', train=True, transform=transform, 
        num_patches=8, patch_size=8, mask_value=0.0
    )
    
    # Get a sample
    original_image, masked_image, label = masked_mnist_dataset[0]
    print(f"Image shape: {original_image.shape}, Label: {label}")
    print(f"Number of patches masked: {masked_mnist_dataset.num_patches}")
    print(f"Patch size: {masked_mnist_dataset.patch_size}x{masked_mnist_dataset.patch_size}")
    print(f"Mask value: {masked_mnist_dataset.mask_value}")
    
    # Visualize the image
    import matplotlib.pyplot as plt
    
    # Create subplot for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot original image
    ax1.imshow(original_image.squeeze(), cmap='gray')
    ax1.set_title(f'Original - Label: {label}')
    ax1.axis('off')
    
    # Plot masked image
    ax2.imshow(masked_image.squeeze(), cmap='gray')
    ax2.set_title(f'Masked ({masked_mnist_dataset.num_patches} patches of {masked_mnist_dataset.patch_size}x{masked_mnist_dataset.patch_size}) - Label: {label}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show multiple samples to see the randomness
    print("\nShowing multiple samples with different random masks:")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # Get the same image multiple times to show different random masks
        original, masked, label = masked_mnist_dataset[0]
        
        # Plot original (only in first row)
        if i == 0:
            axes[0, i].imshow(original.squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original\nLabel: {label}')
        else:
            axes[0, i].imshow(original.squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Plot masked version
        axes[1, i].imshow(masked.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Masked #{i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Random Masking Examples: {masked_mnist_dataset.num_patches} patches of {masked_mnist_dataset.patch_size}x{masked_mnist_dataset.patch_size}')
    plt.tight_layout()
    plt.show()
