import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from unet_with_time import UNet  # Assuming the UNet model is defined in unet_with_time.py
import matplotlib.pyplot as plt
import numpy as np
import math

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T_RANGE = 100

def scheduler(t: float):
    # Use a more stable scheduler that doesn't cause overflow
    # Linear schedule from 0.0001 to 0.02
    beta_start = 0.0001
    beta_end = 0.02
    return beta_start + (beta_end - beta_start) * (t / T_RANGE)

device = "mps"
alpha_bar_memo = {}

# Compute alpha_bar values more carefully to avoid overflow
for t in range(T_RANGE):
    beta_t = scheduler(t)
    alpha_t = 1.0 - beta_t
    
    if t == 0:
        alpha_bar_memo[t] = alpha_t
    else:
        alpha_bar_memo[t] = alpha_t * alpha_bar_memo[t - 1]
    
    # Clamp to prevent numerical issues
    alpha_bar_memo[t] = max(alpha_bar_memo[t], 1e-8)

# print first few alpha_bar values
print("Alpha bar values:", list(alpha_bar_memo.values())[:10])

def add_noise(images, t_tensor):
    """Add random noise to images for denoising task"""
    # Create a copy to avoid modifying the original tensor
    t_copy = t_tensor.clone().to("cpu")
    
    # Get alpha_bar values for each timestep
    alpha_bar_values = []
    for t_val in t_copy:
        alpha_bar_values.append(alpha_bar_memo[t_val.item()])
    
    alpha_bar = torch.tensor(alpha_bar_values, device=device)
    alpha_bar = alpha_bar.view(-1, 1, 1, 1)  # Reshape for broadcasting
    
    noise = torch.randn_like(images) 
    noisy_images = (alpha_bar ** 0.5) * images + noise * ((1 - alpha_bar) ** 0.5)
    return torch.clamp(noisy_images, 0., 1.), noise

def train_model(model, train_loader, val_loader, epochs=3, lr=0.001):
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            t_values = torch.randint(0, T_RANGE, (data.size(0),), device=device)
            
            # Create noisy input and clean target
            noisy_data, noise = add_noise(data, t_values)
            optimizer.zero_grad()
            t_values = t_values.float() / (T_RANGE - 1)  # Normalize to [0, 1]
            output = model(noisy_data, t_values)
            loss = criterion(output, noise)
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
                t_values = torch.randint(0, T_RANGE, (data.size(0),), device=device)
                noisy_data, noise = add_noise(data, t_values)
                t_values = t_values.float() / (T_RANGE - 1)  # Normalize to [0, 1]
                output = model(noisy_data, t_values)
                val_loss += criterion(output, noise).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def visualize_results(model, test_loader, num_samples=3):
    """Visualize the diffusion process: forward noising and reverse denoising"""
    model.eval()
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        
        # Define timesteps to visualize
        forward_timesteps = [i for i in range(0, T_RANGE, T_RANGE // 5)] + [T_RANGE - 1]  # Forward diffusion timesteps
        reverse_timesteps = forward_timesteps[::-1]
        
        # Create figure for forward diffusion process
        fig1, axes1 = plt.subplots(num_samples, len(forward_timesteps), figsize=(15, 8))
        if num_samples == 1:
            axes1 = axes1.reshape(1, -1)
        
        # Forward diffusion process
        for sample_idx in range(num_samples):
            sample = data[sample_idx:sample_idx+1]  # Keep batch dimension
            
            for t_idx, t in enumerate(forward_timesteps):
                t_tensor = torch.tensor([t], device=device)
                noisy_sample, _ = add_noise(sample, t_tensor)
                
                # Display the noisy image
                img = noisy_sample.cpu().squeeze()
                axes1[sample_idx, t_idx].imshow(img, cmap='gray')
                axes1[sample_idx, t_idx].set_title(f't={t}')
                axes1[sample_idx, t_idx].axis('off')
        
        plt.suptitle('Forward Diffusion Process (Adding Noise)', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Create figure for reverse diffusion process (denoising)
        fig2, axes2 = plt.subplots(num_samples, len(reverse_timesteps), figsize=(15, 8))
        if num_samples == 1:
            axes2 = axes2.reshape(1, -1)
        
        # Reverse diffusion process
        for sample_idx in range(num_samples):
            sample = data[sample_idx:sample_idx+1]  # Keep batch dimension
            
            # Use the abstracted reverse diffusion function
            final_sample, display_images = reverse_diffusion_sampling(
                model, 
                sample.shape, 
                device, 
                store_intermediate=True, 
                store_timesteps=reverse_timesteps
            )
            
            # Display the stored images
            for t_idx, t in enumerate(reverse_timesteps):
                img = display_images[t]
                axes2[sample_idx, t_idx].imshow(img, cmap='gray')
                axes2[sample_idx, t_idx].set_title(f't={t}')
                axes2[sample_idx, t_idx].axis('off')
        
        plt.suptitle('Reverse Diffusion Process (Denoising)', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Show comparison: Original vs Final Denoised
        fig3, axes3 = plt.subplots(2, num_samples, figsize=(12, 6))
        if num_samples == 1:
            axes3 = axes3.reshape(2, 1)
        
        for sample_idx in range(num_samples):
            # Original
            axes3[0, sample_idx].imshow(data[sample_idx].cpu().squeeze(), cmap='gray')
            axes3[0, sample_idx].set_title('Original')
            axes3[0, sample_idx].axis('off')
            
            # Final denoised result using the abstracted reverse diffusion function
            sample = data[sample_idx:sample_idx+1]
            current_sample = reverse_diffusion_sampling(
                model, 
                sample.shape, 
                device, 
                store_intermediate=False
            )

            axes3[1, sample_idx].imshow(current_sample.cpu().squeeze(), cmap='gray')
            axes3[1, sample_idx].set_title('Generated')
            axes3[1, sample_idx].axis('off')
        
        plt.suptitle('Original vs Generated Images', fontsize=16)
        plt.tight_layout()
        plt.show()

def filter_dataset_by_label(dataset, target_label):
    """Filter dataset to only include samples with specific label"""
    filtered_indices = []
    for idx, (_, label) in enumerate(dataset):
        if label == target_label:
            filtered_indices.append(idx)
    return torch.utils.data.Subset(dataset, filtered_indices)

def reverse_diffusion_sampling(model, sample_shape, device, store_intermediate=False, store_timesteps=None):
    """
    Perform full reverse diffusion sampling from pure noise to clean image.
    
    Args:
        model: The trained UNet model
        sample_shape: Shape of the sample to generate (e.g., (1, 1, 28, 28))
        device: Device to run on
        store_intermediate: Whether to store intermediate results
        store_timesteps: List of timesteps to store (only used if store_intermediate=True)
    
    Returns:
        If store_intermediate=True: (final_sample, intermediate_images_dict)
        If store_intermediate=False: final_sample
    """
    # Start with pure noise
    current_sample = torch.randn(sample_shape, device=device)
    
    # Store intermediate results if requested
    intermediate_images = {} if store_intermediate else None
    
    # Perform full denoising process from T_RANGE-1 to 0
    for t in range(T_RANGE-1, -1, -1):
        if store_intermediate and store_timesteps and t in store_timesteps:
            # Store current image for display
            intermediate_images[t] = current_sample.cpu().squeeze().clone()
        
        if t > 0:  # Don't denoise at t=0, it's the final result
            # Denoise step by step
            t_tensor = torch.tensor([t], device=device)
            t_normalized = t_tensor.float() / (T_RANGE - 1)
            
            # Predict noise
            predicted_noise = model(current_sample, t_normalized)
            
            # Remove predicted noise (DDPM sampling step)
            alpha_bar = alpha_bar_memo[t]
            alpha_bar_prev = alpha_bar_memo[t-1]
            alpha = alpha_bar / alpha_bar_prev
            
            # Clamp alpha_bar values to prevent overflow
            alpha_bar = max(alpha_bar, 1e-8)
            alpha = max(alpha, 1e-8)
            
            # DDPM sampling step
            noise = torch.randn_like(current_sample)
            # Use stable formulation
            sqrt_alpha_bar = alpha_bar ** 0.5
            sqrt_alpha = alpha ** 0.5
            sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
            sqrt_one_minus_alpha = (1 - alpha) ** 0.5
            
            current_sample = (1 / sqrt_alpha) * (current_sample - 
                            ((1 - alpha) / (sqrt_one_minus_alpha_bar)) * predicted_noise) + \
                            sqrt_one_minus_alpha * noise
            
            current_sample = torch.clamp(current_sample, 0., 1.)
    
    # Store final result at t=0 if storing intermediate results
    if store_intermediate:
        intermediate_images[0] = current_sample.cpu().squeeze().clone()
        return current_sample, intermediate_images
    else:
        return current_sample

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
    # train_dataset_filtered = filter_dataset_by_label(train_dataset, target_label=3)
    train_dataset_filtered = train_dataset
    # test_dataset_filtered = filter_dataset_by_label(test_dataset, target_label=3)
    test_dataset_filtered = test_dataset
    
    print(f"Original train dataset size: {len(train_dataset)}")
    print(f"Filtered train dataset size (label 1 only): {len(train_dataset_filtered)}")
    print(f"Original test dataset size: {len(test_dataset)}")
    print(f"Filtered test dataset size (label 1 only): {len(test_dataset_filtered)}")
    batch_size = 1024
    # Create data loaders
    test_loader = DataLoader(test_dataset_filtered, batch_size=batch_size, shuffle=False)
    
    # Split filtered train into train/val
    train_size = int(0.8 * len(train_dataset_filtered))
    val_size = len(train_dataset_filtered) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset_filtered, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    print(f"Train loader size: {len(train_loader.dataset)}")
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    print(f"Validation loader size: {len(val_loader.dataset)}")
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1, embedding_dim=64)
    print(model)
    print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train the model
    print('Starting training...')
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=30)
    
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
