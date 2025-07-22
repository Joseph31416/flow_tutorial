"""
Takes a tensor of shape (B, C, H, W) and returns a tensor of shape
(B, C, n_patches, patch_size, patch_size) where n_patches = (H // patch_size) * (W // patch_size).
This is useful for image tokenization in vision transformers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def tokenize_image(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Tokenizes an image tensor into patches.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        patch_size (int): Size of each patch.

    Returns:
        torch.Tensor: Tokenized image tensor of shape (B, C, n_patches, patch_size, patch_size).
    """
    B, C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Height and width must be divisible by patch_size."
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    n_patches = n_patches_h * n_patches_w
    patches = image.unfold(2, patch_size, patch_size)
    patches = patches.unfold(3, patch_size, patch_size)
    patches = patches.contiguous()
    patches = patches.view(B, C, n_patches, patch_size, patch_size)
    return patches

def tokens_to_image(tokens: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Converts tokenized image patches back to the original image tensor.

    Args:
        tokens (torch.Tensor): Tokenized image tensor of shape (B, C, n_patches, patch_size, patch_size).
        patch_size (int): Size of each patch.

    Returns:
        torch.Tensor: Reconstructed image tensor of shape (B, C, H, W).
    """
    B, C, n_patches, _, _ = tokens.shape
    n_patches_h = int(np.sqrt(n_patches))
    n_patches_w = n_patches // n_patches_h
    H = n_patches_h * patch_size
    W = n_patches_w * patch_size
    tokens = tokens.view(B, C, n_patches_h, n_patches_w, patch_size, patch_size)
    tokens = tokens.permute(0, 1, 2, 4, 3, 5)  # Rearrange to (B, C, n_patches_h, patch_size, n_patches_w, patch_size)
    tokens = tokens.contiguous()
    image = tokens.view(B, C, H, W)  # Reshape to (B, C, H, W)
    return image

if __name__ == "__main__":
    # load mnist dataset
    import random
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    image, _ = mnist_dataset[random.randint(0, len(mnist_dataset) - 1)]
    image = image.unsqueeze(0)  # Add batch dimension (B=1, C=1, H=28, W=28)
    patch_size = 7  # Example patch size
    tokenized_image = tokenize_image(image, patch_size)
    print(f"Original image shape: {image.shape}")
    print(f"Tokenized image shape: {tokenized_image.shape}")
    # Visualize the tokenized patches
    patches = tokenized_image[0, 0]  # Get the first image's
    patches = patches.view(-1, patch_size, patch_size)  # Reshape to (n_patches, patch_size, patch_size)
    n_patches = patches.shape[0]
    # plot in 7 x 7 grid
    grid_size = int(np.ceil(np.sqrt(n_patches)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < n_patches:
            ax.imshow(patches[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # show conversion back to image
    reconstructed_image = tokens_to_image(tokenized_image, patch_size)
    print(f"Reconstructed image shape: {reconstructed_image.shape}")
    plt.imshow(reconstructed_image[0, 0].squeeze(), cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.show()
                                                            

