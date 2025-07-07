import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from load_utils import load_data
from mnist_autoencoder_cnn import CNNAutoencoder
import tqdm
import os

class Flow(nn.Module):
    def __init__(self, dim: int = 784, h: int = 512):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(dim + 1 + 10, h),
        nn.ELU(),
        nn.Linear(h, h),
        nn.ELU(),
        nn.Linear(h, h),
        nn.ELU(),
        nn.Linear(h, dim),
        nn.Sigmoid()  # Sigmoid to keep output in [0, 1] range
    )
    def forward(self, x_t: Tensor, c: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t, c, x_t), -1))
    def step(self, x_t: Tensor, c: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + (t_end - t_start) * self(
            x_t + self(x_t, c, t_start) * (t_end - t_start) / 2, c,
            t_start + (t_end - t_start) / 2)
    
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")
# training
latent_dim = 16  # Set the latent dimension for the autoencoder
flow = Flow(dim=latent_dim).to(device)
autoencoder = CNNAutoencoder(latent_dim=latent_dim).to(device)
# load weights
autoencoder_weight_path = os.path.join("mnist", "models", "best_cnn_autoencoder.pth")
if os.path.exists(autoencoder_weight_path):
    autoencoder.load_state_dict(torch.load(autoencoder_weight_path, map_location=device))
    print("Autoencoder weights loaded successfully.")

# print model
print(flow)
# print number of parameters
print(f'Number of parameters: {sum(p.numel() for p in flow.parameters() if p.requires_grad)}')
# use AdamW optimizer with learning rate 1e-3
optimizer = torch.optim.AdamW(flow.parameters(), 1e-3)
loss_fn = nn.MSELoss()
num_epochs = 500
batch_size = 4096

# Initialize loss history lists
train_loss_history = []
val_loss_history = []

train_loader, val_loader, test_loader = load_data(batch_size=batch_size)
for epoch in tqdm.tqdm(range(num_epochs)):
    train_loss = 0.0
    flow.train()
    for data, labels in train_loader:
        # one-hot encode labels to use as condition
        labels_one_hot = torch.nn.functional.one_hot(
            labels, num_classes=10
        ).float().to(device)
        x_1 = autoencoder.encoder(data.to(device))
        # concat the one-hot encoded labels to the encoded data
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(len(x_1), 1).to(device)
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        optimizer.zero_grad()
        loss = loss_fn(flow(x_t, labels_one_hot, t), dx_t)
        loss.backward() 
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
    
    # Validation
    flow.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, labels in val_loader:
            # flatten the images from (batch_size, 1, 28, 28) to (batch_size, 784)
            labels_one_hot = torch.nn.functional.one_hot(
                labels, num_classes=10
            ).float().to(device)
            x_1 = autoencoder.encoder(data.to(device))
            # concat the one-hot encoded labels to the encoded data
            x_0 = torch.randn_like(x_1).to(device)
            t = torch.rand(len(x_1), 1).to(device)
            x_t = (1 - t) * x_0 + t * x_1
            dx_t = x_1 - x_0
            val_loss += loss_fn(flow(x_t, labels_one_hot, t), dx_t).item()
    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}', flush=True)

# Save the trained model
torch.save(flow.state_dict(), 'flow_model.pth')
print("Model saved as 'flow_model.pth'")

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_history.png', dpi=300, bbox_inches='tight')
plt.close()

# sampling - generate multiple images
flow.eval()
n_samples = 4
samples_labels = torch.randint(0, 10, (n_samples,)).to(device)  # Random labels for sampling
n_steps = 8
fig, axes = plt.subplots(n_samples, n_steps + 1, figsize=(20, 8))
if n_samples == 1:
    axes = axes.reshape(1, -1)

time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)

for sample_idx in range(n_samples):
    x = torch.randn(1, latent_dim).to(device)  # MNIST images are 28x28 = 784
    decoded_x = autoencoder.decode(x)  # Reshape to (batch_size, 784)
    decoded_x = decoded_x.view(decoded_x.size(0), -1)  # Flatten to (batch_size, 784)
    # plot the generated image across the time steps
    axes[sample_idx, 0].imshow(decoded_x.detach().cpu().numpy().reshape(28, 28), cmap='gray')
    axes[sample_idx, 0].set_title(f'Label: {samples_labels[sample_idx].item()}, t = {time_steps[0]:.2f}')
    axes[sample_idx, 0].axis('off')
    
    for i in range(n_steps):
        c = torch.nn.functional.one_hot(
            samples_labels[sample_idx], num_classes=10
        ).float().to(device).unsqueeze(0)
        x = flow.step(x, c, time_steps[i], time_steps[i + 1])
        decoded_x = autoencoder.decode(x)
        decoded_x = decoded_x.view(decoded_x.size(0), -1)
        axes[sample_idx, i + 1].imshow(decoded_x.detach().cpu().numpy().reshape(28, 28), cmap='gray')
        axes[sample_idx, i + 1].set_title(f'Label: {samples_labels[sample_idx].item()}, t = {time_steps[i + 1]:.2f}')
        axes[sample_idx, i + 1].axis('off')

plt.tight_layout()
plt.savefig('flow_samples.png', dpi=300, bbox_inches='tight')
plt.close()
