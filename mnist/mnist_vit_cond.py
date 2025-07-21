import torch
import torch.nn as nn
from vit_cond import PatchEmbed, ViTEncoder, ViTDecoder
from copy import deepcopy
import tqdm
import torchvision
import matplotlib.pyplot as plt

DEVICE = "mps"
BETA = 1

class ConditionalViTVAE(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        embed_dim = kwargs.get('embed_dim', 128)
        # patch + position
        self.patch_embed = PatchEmbed(**kwargs)
        # learnable label embedding
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        # encoder & decoder
        encoder_kwargs = deepcopy(kwargs)
        encoder_kwargs.pop('img_size', None)  # img_size not needed in encoder
        encoder_kwargs.pop('patch_size', None)
        self.encoder     = ViTEncoder(**encoder_kwargs)
        self.decoder     = ViTDecoder(**kwargs)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def forward(self, x, labels):
        # 1) patchify + position
        patches = self.patch_embed(x)               # [B, n_patches, D]
        # 2) inject label into every token
        le = self.label_embed(labels)               # [B, D]
        patches = patches + le.unsqueeze(1)         # broadcast

        # 3) encode to latent stats
        mu, logvar = self.encoder(patches)

        # 4) reparam
        z = self.reparameterize(mu, logvar)
        # 5) condition z as well
        z = z + le

        # 6) decode
        recon = self.decoder(z)
        return recon, mu, logvar

def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Calculate the KL divergence loss."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Calculate the reconstruction loss."""
    return torch.nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')

def train(model, dataloader, optimizer, curr_epoch: int):
    model.train()
    total_loss = 0
    for images, labels in tqdm.tqdm(dataloader, desc=f'Epoch {curr_epoch}'):
        # Assuming labels are used as conditions
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).long()
        optimizer.zero_grad()
        x_hat, mu, logvar = model(images, labels)
        reconstruction_loss_val = reconstruction_loss(x_hat, images)
        kl = kl_loss(mu, logvar)
        loss = reconstruction_loss_val + kl * BETA
        loss = loss / images.size(0)  # Average loss over batch
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
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()
            x_hat, mu, logvar = model(images, labels)
            reconstruction_loss_val = reconstruction_loss(x_hat, images)
            kl = kl_loss(mu, logvar)
            loss = reconstruction_loss_val + kl * BETA
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Validation Loss: {avg_loss:.4f}')

def training_loop(model: ConditionalViTVAE, dataloader, optimizer, epochs: int = 10):
    for epoch in range(epochs):
        train(model, dataloader, optimizer, curr_epoch=epoch)
        validate(model, dataloader)
        print(f'Epoch {epoch + 1}/{epochs} completed.')
    print('Training complete.')

def visualize_samples(model: ConditionalViTVAE, dataloader, num_samples: int = 5):
    model.eval()
    
    with torch.no_grad():
        # Get a batch of real images and labels
        images, labels = next(iter(dataloader))
        images = images[:num_samples].to(DEVICE)
        labels = labels[:num_samples].to(DEVICE).long()
    
        # Get reconstructions of real images
        reconstructions, _, _ = model(images, labels)
        
        # Move to CPU for visualization
        images = images.cpu()
        reconstructions = reconstructions.cpu()
        labels = labels.cpu()
        
        # Plot original images and their reconstructions
        fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(images[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original (Label: {labels[i].item()})')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Reconstructed')
            axes[1, i].axis('off')
        
        plt.suptitle('Original vs Reconstructed Images')
        plt.tight_layout()
        plt.show()
        
        # Generate samples from random latent vectors with random labels
        embed_dim = 128  # This should match the model's embed_dim
        z = torch.randn(num_samples, embed_dim).to(DEVICE)
        
        # Generate random labels
        random_labels = torch.randint(0, 10, (num_samples,)).to(DEVICE)
        random_cond = torch.nn.functional.one_hot(random_labels, num_classes=10).float()
        
        # Add label embedding to latent vector (as done in the model's forward pass)
        label_embed = model.label_embed(random_labels)
        z_conditioned = z + label_embed
        
        # Generate images from conditioned latent vectors
        generated = model.decoder(z_conditioned)
        
        # Move to CPU for visualization
        generated = generated.cpu()
        random_labels = random_labels.cpu()
        
        # Plot generated samples
        fig, axes = plt.subplots(1, num_samples, figsize=(10, 3))
        for i in range(num_samples):
            axes[i].imshow(generated[i].squeeze(), cmap='gray')
            axes[i].set_title(f'Generated (Label: {random_labels[i].item()})')
            axes[i].axis('off')
        
        plt.suptitle('Generated Samples from Random Latent Vectors')
        plt.tight_layout()
        plt.show()


def main():

    # Load MNIST dataset
    # binarise the images to 0 or 1 based on threshold of 0.5
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Lambda(lambda x: (x > 0.5).float()),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    batch_size = 256
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer
    model = ConditionalViTVAE(
        num_classes=10, img_size=28, patch_size=4, embed_dim=128
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train the model
    training_loop(model, train_dataloader, optimizer, epochs=2)
    # save the model
    torch.save(model.state_dict(), 'mnist_vae_model.pth')

    # load the model
    model.load_state_dict(torch.load('mnist_vae_model.pth', map_location=DEVICE))

    # Visualize samples
    visualize_samples(model, val_dataloader)

if __name__ == "__main__":
    main()
