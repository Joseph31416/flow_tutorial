import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

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

def visualize_forward_diffusion():
    """Visualize the forward diffusion process on MNIST images"""
    
    # Load a sample MNIST image
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Get a few sample images
    num_samples = 3
    sample_indices = [0, 100, 200]  # Different digit examples
    
    # Define timesteps to visualize
    timesteps = [0, 10, 25, 50, 75, 99]
    
    fig, axes = plt.subplots(num_samples, len(timesteps), figsize=(18, 8))
    
    for sample_idx, img_idx in enumerate(sample_indices):
        # Get the image and label
        image, label = dataset[img_idx]
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        
        print(f"Sample {sample_idx + 1}: Digit {label}")
        
        for t_idx, t in enumerate(timesteps):
            # Create tensor for timestep
            t_tensor = torch.tensor([t], device=device)
            
            # Add noise at timestep t
            noisy_image, noise = add_noise(image, t_tensor)
            
            # Convert to numpy for visualization
            img_np = noisy_image.cpu().squeeze().numpy()
            
            # Plot the image
            axes[sample_idx, t_idx].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            axes[sample_idx, t_idx].set_title(f't={t}\nα̅={alpha_bar_memo[t]:.4f}')
            axes[sample_idx, t_idx].axis('off')
    
    plt.suptitle('Forward Diffusion Process: Progressive Noise Addition', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_alpha_schedule():
    """Plot the alpha and alpha_bar schedules"""
    timesteps = list(range(T_RANGE))
    alpha_values = []
    alpha_bar_values = []
    beta_values = []
    
    for t in timesteps:
        beta_t = scheduler(t)
        alpha_t = 1.0 - beta_t
        
        beta_values.append(beta_t)
        alpha_values.append(alpha_t)
        alpha_bar_values.append(alpha_bar_memo[t])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot beta schedule
    axes[0].plot(timesteps, beta_values, 'b-', linewidth=2)
    axes[0].set_title('Beta Schedule (Noise Rate)')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('β_t')
    axes[0].grid(True, alpha=0.3)
    
    # Plot alpha schedule
    axes[1].plot(timesteps, alpha_values, 'g-', linewidth=2)
    axes[1].set_title('Alpha Schedule (1 - β_t)')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('α_t')
    axes[1].grid(True, alpha=0.3)
    
    # Plot alpha_bar schedule
    axes[2].plot(timesteps, alpha_bar_values, 'r-', linewidth=2)
    axes[2].set_title('Alpha Bar Schedule (Cumulative)')
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('α̅_t')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')  # Log scale to better see the decay
    
    plt.tight_layout()
    plt.show()

def demonstrate_noise_levels():
    """Demonstrate how noise level changes with different timesteps"""
    
    # Load a single MNIST image
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    image, label = dataset[7]  # Pick a "7" digit
    image = image.unsqueeze(0).to(device)
    
    print(f"Demonstrating noise levels on digit: {label}")
    
    # Create a range of timesteps
    timesteps = list(range(0, T_RANGE, 10))  # Every 10 timesteps
    
    num_cols = 10
    num_rows = len(timesteps) // num_cols + (1 if len(timesteps) % num_cols else 0)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 2 * num_rows))
    axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes
    
    for i, t in enumerate(timesteps):
        if i < len(axes):
            t_tensor = torch.tensor([t], device=device)
            noisy_image, _ = add_noise(image, t_tensor)
            img_np = noisy_image.cpu().squeeze().numpy()
            
            axes[i].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f't={t}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(timesteps), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Noise Progression on Digit {label}', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting forward diffusion visualization...")
    
    # Show the schedule plots
    print("1. Plotting noise schedules...")
    plot_alpha_schedule()
    
    # Show forward diffusion on sample images
    print("2. Visualizing forward diffusion process...")
    visualize_forward_diffusion()
    
    # Show detailed noise progression
    print("3. Demonstrating noise level progression...")
    demonstrate_noise_levels()
    
    print("Visualization complete!")

