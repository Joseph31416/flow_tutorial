import torch
import torch.nn as nn
from img_tokenizer import tokenize_image, tokens_to_image

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # x = self.norm(x)
        x = self.relu(x)
        return x
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # x = self.norm(x)
        x = self.relu(x)
        return x
    
class MultiHeadAttentionOptimised(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_heads: int, p: float = 0.1):
        super(MultiHeadAttentionOptimised, self).__init__()
        self.c_attn = nn.Linear(in_channels, 3 * out_channels, bias=False)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=-1)
        self.n_heads = num_heads
        self.d = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assume x is of shape (B, N, C) where B is batch size, N is number of patches, and C is channels.
        Returns the attention output of shape (B, N, out_channels).
        """
        c_attn = self.c_attn(x)  # (B, N, 3 * out_channels)
        q, k, v = c_attn.chunk(3, dim=-1) # each is (B, N, out_channels)
        q = q.view(q.size(0), q.size(1), self.n_heads, -1).transpose(1, 2)  # (B, n_heads, N, d)
        k = k.view(k.size(0), k.size(1), self.n_heads, -1).transpose(1, 2)  # (B, n_heads, N, d)
        v = v.view(v.size(0), v.size(1), self.n_heads, -1).transpose(1, 2)  # (B, n_heads, N, d)
        k_t = k.transpose(-2, -1)  # Transpose for matrix multiplication
        attn_score = self.softmax(torch.matmul(q, k_t) / (self.d ** 0.5)) # (B, n_heads, N, N)
        attn_output = torch.matmul(attn_score, v)  # (B, n_heads, N, d)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N, n_heads, d)
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)
        attn_output = self.dropout(attn_output)  # (B, N, out_channels)
        return attn_output

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_size = d_model // 2
        pe[:, 1::2] = torch.cos(position * div_term)[:, :cos_size]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assume x is of shape (B, N, C).
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class ViMHA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int, patch_size: int):
        super(ViMHA, self).__init__()
        self.mha = MultiHeadAttentionOptimised(
            in_channels * patch_size * patch_size,
            out_channels * patch_size * patch_size,
            num_heads
        )
        self.positional_encoding = SinusoidalPositionalEncoding(in_channels * patch_size * patch_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assume x is of shape (B, C, H, W). Tokenize it to (B, C, n_patches, patch_size, patch_size),
        then apply multi-head attention.
        """
        x = tokenize_image(x, self.patch_size)  # Tokenize to patches (B, C, n_patches, patch_size, patch_size)
        B, C, n_patches, patch_size, _ = x.shape
        x = x.view(B, n_patches, C * patch_size * patch_size)
        pe = self.positional_encoding(x)
        x = pe + x  # Add positional encoding
        x = self.mha(x)  # Apply multi-head attention (B, n_patches, C * patch_size * patch_size)
        x = x.view(B, C, n_patches, patch_size, patch_size)
        x = tokens_to_image(x, self.patch_size)  # Convert back to image patches
        return x

class DownBlockWithResAttn(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int, patch_size: int, height: int, width: int):
        super(DownBlockWithResAttn, self).__init__()
        self.downsample = DownSampleBlock(in_channels, out_channels)
        dim = (height // 2) * (width // 2)
        self.ln = nn.LayerNorm(dim)
        self.mha = ViMHA(out_channels, out_channels, num_heads, patch_size)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be of shape (B, C, H, W).
        """
        x = self.downsample(x)
        B, C, H, W = x.shape
        x = x.view(x.size(0) * x.size(1), -1)
        x = self.ln(x)
        x = x.view(B, C, H, W)  # Reshape back to (B, out_channels, H, W)
        x = x + self.mha(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # Flatten to (B, C, H * W)
        x = self.linear(x)  # Apply linear transformation
        x = x.view(B, C, H, W)
        return x
    
class UpBlockWithResAttn(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int, patch_size: int, height: int, width: int):
        super(UpBlockWithResAttn, self).__init__()
        self.upsample = UpSampleBlock(in_channels, out_channels)
        dim = (height * 2) * (width * 2)
        self.ln = nn.LayerNorm(dim)
        self.mha = ViMHA(out_channels, out_channels, num_heads, patch_size)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be of shape (B, C, H, W).
        """
        x = self.upsample(x)
        B, C, H, W = x.shape
        x = x.view(x.size(0) * x.size(1), -1)
        x = self.ln(x)
        x = x.view(B, C, H, W)  # Reshape back to (B, out_channels, H, W)
        x = x + self.mha(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # Flatten to (B, C, H * W)
        x = self.linear(x)  # Apply linear transformation
        x = x.view(B, C, H, W)
        return x

class ViTUnetResNorm(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_heads: int, patch_size: int):
        super(ViTUnetResNorm, self).__init__()
        self.db_1 = DownBlockWithResAttn(in_channels, 16, num_heads, patch_size, 28, 28)
        self.db_2 = DownBlockWithResAttn(16, 32, num_heads, patch_size, 14, 14)
        self.up_1 = UpBlockWithResAttn(32, 16, num_heads, patch_size, 7, 7)
        self.up_2 = UpBlockWithResAttn(16, out_channels, 7, patch_size, 14, 14) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be of shape (B, 1, 28, 28) for MNIST-like images.
        """
        x = self.db_1(x)
        x = self.db_2(x)
        x = self.up_1(x)
        x = self.up_2(x)
        return x
        
class VAEViTUnetResNorm(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int, patch_size: int, latent_dim: int = 32):
        super(VAEViTUnetResNorm, self).__init__()
        self.latent_dim = latent_dim

        self.db_1 = DownBlockWithResAttn(in_channels, 64, num_heads, patch_size, 28, 28)
        self.db_2 = DownBlockWithResAttn(64, 128, num_heads, patch_size, 14, 14)
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 7 * 7)
        self.sigmoid = nn.Sigmoid()

        self.up_1 = UpBlockWithResAttn(128, 64, num_heads, patch_size, 7, 7)
        self.up_2 = UpBlockWithResAttn(64, out_channels, 7, patch_size, 14, 14)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode the input tensor into mean and log variance.
        """
        x = self.db_1(x)
        x = self.db_2(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent vector back to the original image space.
        """
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.up_1(x)
        x = self.up_2(x)
        # x = self.sigmoid(x)  # Ensure output is in the range [0, 1]
        return x
    
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from the VAE.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be of shape (B, 1, 28, 28) for MNIST-like images.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

if __name__ == "__main__":
    # Example usage
    B, C, H, W = 4, 1, 28, 28
    patch_size = 7
    x = torch.randn(B, 49, 16)  # Random input tensor
    # model = ViTUnet(in_channels=C, out_channels=1, num_heads=4, patch_size=patch_size)
    # output = model(x)
    # print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    mha_optimised = MultiHeadAttentionOptimised(in_channels=16, out_channels=16, n_heads=4)
    output = mha_optimised(x)  # Flatten the input
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")