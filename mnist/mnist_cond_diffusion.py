import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from unet_with_time_cond import UNet
from typing import Union
import torchvision
import tqdm

T_RANGE = 100
DEVICE = "mps"  # Change to "cuda" if using a GPU

class Scheduler:

    def __init__(self, t_range: int):
        self.t_range = t_range
        self.alpha_values = []
        self.alpha_bar_values = []

    def __call__(self, t: Union[int, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            # it is a tensor of shape (N, )
            # return the alpha and alpha_bar values for each element in the tensor\
            t = t.detach().cpu().numpy().tolist()
            alpha_val_tensor = torch.tensor([self.alpha_values[t_i] for t_i in t], dtype=torch.float32)
            alpha_bar_val_tensor = torch.tensor([self.alpha_bar_values[t_i] for t_i in t], dtype=torch.float32)
            return alpha_val_tensor, alpha_bar_val_tensor
        return self.alpha_values[t], self.alpha_bar_values[t]

class LinearScheduler(Scheduler):
    
    def __init__(self, t_range: int, alpha_start: float = 0.999, alpha_end: float = 0.9):
        super().__init__(t_range)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self._compute_alpha_values()

    def _compute_alpha_values(self):
        for t in range(self.t_range):
            alpha_t = self.alpha_start + (self.alpha_end - self.alpha_start) * (t / (self.t_range - 1))
            self.alpha_values.append(alpha_t)
            if t == 0:
                self.alpha_bar_values.append(alpha_t)
            else:
                self.alpha_bar_values.append(alpha_t * self.alpha_bar_values[t - 1])
        
        # Clamp to prevent numerical issues
        self.alpha_bar_values = [max(val, 1e-8) for val in self.alpha_bar_values]

class DiffusionModel(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 t_emb_dim: int, cond_emb_dim: int, device: str = DEVICE):
        super(DiffusionModel, self).__init__()
        self.unet = UNet(in_channels, out_channels, t_emb_dim, cond_emb_dim).to(device)
        self.device = device
        # print device for unet
        print(f"UNet initialized on device: {device}")
        self.scheduler = LinearScheduler(t_range=T_RANGE)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        # Get alpha and alpha_bar values from the scheduler
        x = x.to(self.device)  # Ensure x is on the correct device
        alpha, alpha_bar = self.scheduler(t)
        
        alpha_bar = alpha_bar.view(-1, 1, 1, 1).to(self.device)  # Reshape for broadcasting
        alpha = alpha.view(-1, 1, 1, 1).to(self.device)  # Reshape for broadcasting
        eps = torch.randn_like(x).to(self.device)  # Random noise
        z = (alpha_bar ** 0.5) * x + eps * ((1 - alpha_bar) ** 0.5)
        z = torch.clamp(z, 0, 1)  # Clamp z to [0, 1]

        # Forward pass through UNet
        t = t / (self.scheduler.t_range - 1)  # Normalize t to [0, 1]
        t = t.to(self.device)  # Ensure t is on the correct device
        cond = cond.to(self.device)  # Ensure cond is on the correct device
        z = z.to(self.device)  # Ensure z is on the correct device
        output = self.unet(z, t, cond)
        
        # Return the output and noise for loss computation
        return output, eps
    
    def reverse(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        # Get alpha_bar values for reverse diffusion
        alpha, alpha_bar = self.scheduler(t)
        alpha_bar = alpha_bar.view(-1, 1, 1, 1).to(self.device)  
        alpha = alpha.view(-1, 1, 1, 1).to(self.device)
        noise = torch.randn_like(x).to(self.device)  # Random noise for reverse diffusion
        x = x.to(self.device)  # Ensure x is on the correct device
        t = t.to(self.device)  # Ensure t is on the correct device
        cond = cond.to(self.device)  # Ensure cond is on the correct device
        eps = self.unet(x, t, cond)
        # x = 1 / (alpha ** 0.5) * (x - eps * ((1 - alpha) / (1 - alpha_bar) ** 0.5)) + (1 - alpha) * noise
        x = 1 / (alpha ** 0.5) * (x - eps * ((1 - alpha) / (1 - alpha_bar) ** 0.5))
        # clamp x to [0, 1]
        x = torch.clamp(x, 0, 1)
        return x

def train(model, dataloader, optimizer, curr_epoch: int):
    model.train()
    total_loss = 0
    for images, labels in tqdm.tqdm(dataloader, desc=f'Epoch {curr_epoch}'):
        # Assuming labels are used as conditions
        t = torch.randint(0, T_RANGE, (images.size(0),), device=images.device) 
        cond = torch.nn.functional.one_hot(labels, num_classes=10).float() 
        
        optimizer.zero_grad()
        output, noise = model(images, t, cond)
        loss = torch.mean((output - noise) ** 2)  # Mean Squared Error Loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Average Loss: {avg_loss:.4f}')

def validate(model, dataloader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, labels in dataloader:
            t = torch.randint(0, T_RANGE, (images.size(0),), device=images.device)
            cond = torch.nn.functional.one_hot(labels, num_classes=10).float() 
            output, noise = model(images, t, cond)
            loss = torch.mean((output - noise) ** 2)
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Validation Loss: {avg_loss:.4f}')

def training_loop(model: DiffusionModel, dataloader, optimizer, epochs: int = 10):
    for epoch in range(epochs):
        train(model, dataloader, optimizer, curr_epoch=epoch)
        validate(model, dataloader)
        print(f'Epoch {epoch + 1}/{epochs} completed.')
    print('Training complete.')

def visualize_samples(model: DiffusionModel, dataloader, num_samples: int = 5):
    # show the forward noising process of N randomly selected images
    # then show the denoising process of the same images
    model.eval()
    # pick N random images from the dataloader
    images, labels = next(iter(dataloader))
    indices = torch.randperm(images.size(0))[:num_samples]
    images = images[indices]
    labels = labels[indices]
    t_values = [0, 25, 50, 75, 99]  # Example timesteps to visualize
    fig, axes = plt.subplots(5, 5, figsize=(15, 6))
    with torch.no_grad():
        # forward noising process
        for i, t in enumerate(t_values):
            if i == 0:
                z = images.cpu().numpy().squeeze()
            else:
                _, alpha_bar = model.scheduler(t)
                eps = torch.randn_like(images)  # Random noise
                z = (alpha_bar ** 0.5) * images + eps * ((1 - alpha_bar) ** 0.5)
                z = torch.clamp(z, 0, 1)  # Clamp to [0, 1]
                # plot the noisy images
                # Plot original image in first row
            for row in range(num_samples):
                axes[row, i].imshow(z[row].squeeze(), cmap='gray')
                if i == 0:
                    axes[row, i].set_title(f't = {t_values[i]}, Label: {labels[row].item()}')
                else:
                    axes[row, i].set_title(f't = {t_values[i]}')
                axes[row, i].axis('off')
        plt.tight_layout()
        plt.show()

        t_values_denoise = [T_RANGE - i for i in range(1, T_RANGE + 1)]
        z = torch.rand_like(images)  # Start with random noise for the denoising process
        z = z.to(model.device)  # Ensure z is on the correct device
        # denoising process
        cnt = 4
        fig, axes = plt.subplots(5, 5, figsize=(15, 6))
        for i, t in enumerate(t_values_denoise):
            timesteps = torch.tensor([t] * num_samples)
            cond = torch.nn.functional.one_hot(labels, num_classes=10).float()
            z = model.reverse(z, timesteps, cond)
            z = torch.clamp(z, 0, 1)
            # Plot denoised images
            if t in t_values:
                for row in range(num_samples):
                    axes[row, cnt].imshow(z[row].detach().cpu().squeeze(), cmap='gray')
                    axes[row, cnt].set_title(f't = {t}')
                    axes[row, cnt].axis('off')
                cnt -= 1
        plt.tight_layout()
        plt.show()

def main():

    # Load MNIST dataset
    # binarise the images to 0 or 1 based on threshold of 0.5
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: (x > 0.5).float()),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    batch_size = 256
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer
    model = DiffusionModel(in_channels=1, out_channels=1, t_emb_dim=64, cond_emb_dim=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train the model
    training_loop(model, train_dataloader, optimizer, epochs=5)
    # save the model
    torch.save(model.state_dict(), 'mnist_diffusion_model.pth')

    # load the model
    model.load_state_dict(torch.load('mnist_diffusion_model.pth', map_location=DEVICE))

    # Visualize samples
    visualize_samples(model, val_dataloader)

if __name__ == "__main__":
    main()
