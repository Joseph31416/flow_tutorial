import torch
import torch.nn as nn
from img_tokenizer import tokenize_image, tokens_to_image

def down_sampling_calc(H: int, W: int, kernel_size: int, stride: int, padding: int = 0) -> tuple:
    """
    Calculate the output height and width after down-sampling.

    Args:
        H (int): Input height.
        W (int): Input width.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int, optional): Padding applied to the input. Defaults to 0.

    Returns:
        tuple: Output height and width after down-sampling.
    """
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    return out_H, out_W

def up_sampling_calc(H: int, W: int, kernel_size: int, stride: int, padding: int = 0) -> tuple:
    """
    Calculate the output height and width after up-sampling.

    Args:
        H (int): Input height.
        W (int): Input width.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int, optional): Padding applied to the input. Defaults to 0.

    Returns:
        tuple: Output height and width after up-sampling.
    """
    out_H = (H - 1) * stride - 2 * padding + kernel_size
    out_W = (W - 1) * stride - 2 * padding + kernel_size
    return out_H, out_W

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
    
class AttentionHead(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, p: float = 0.1):
        super(AttentionHead, self).__init__()
        self.q_linear = nn.Linear(in_channels, out_channels)
        self.k_linear = nn.Linear(in_channels, out_channels)
        self.v_linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=-1)
        self.d = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assume x is of shape (B, N, C) where B is batch size, N is number of patches, and C is channels.
        """
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        k_t = k.transpose(-2, -1)  # Transpose for matrix multiplication
        attn_score = self.softmax(torch.matmul(q, k_t) / (self.d ** 0.5))
        attn_output = torch.matmul(attn_score, v)
        attn_output = self.dropout(attn_output)
        return attn_output
    
class MultiHeadAttention(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(in_channels, out_channels // num_heads) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assume x is of shape (B, N, C) where B is batch size, N is number of patches, and C is channels.
        """
        head_outputs = [head(x) for head in self.heads]
        return torch.cat(head_outputs, dim=-1)
    
class MultiHeadAttentionOptimised(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_heads: int, p: float = 0.1):
        super(MultiHeadAttentionOptimised, self).__init__()
        self.c_attn = nn.Linear(in_channels, 3 * out_channels)
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

class ViTUnet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_heads: int, patch_size: int):
        super(ViTUnet, self).__init__()
        self.down1 = DownSampleBlock(in_channels, 16)
        self.mha_down1 = ViMHA(16, 16, num_heads, patch_size)
        self.down2 = DownSampleBlock(16, 32)
        self.mha_down2 = ViMHA(32, 32, num_heads, patch_size)
        self.up1 = UpSampleBlock(32, 16)
        self.mha_up1 = ViMHA(16, 16, num_heads, patch_size)
        self.up2 = UpSampleBlock(16, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.mha_down1(x)
        x = self.down2(x)
        x = self.mha_down2(x)
        x = self.up1(x)
        x = self.mha_up1(x)
        x = self.up2(x)
        return x

class ViTUnetResNorm(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_heads: int, patch_size: int):
        super(ViTUnetResNorm, self).__init__()
        self.down1 = DownSampleBlock(in_channels, 16)
        self.ln_down1 = nn.LayerNorm(14 * 14)
        self.mha_down1 = ViMHA(16, 16, num_heads, patch_size)
        self.down2 = DownSampleBlock(16, 32)
        self.ln_down2 = nn.LayerNorm(7 * 7)
        self.mha_down2 = ViMHA(32, 32, num_heads, patch_size)
        self.up1 = UpSampleBlock(32, 16)
        self.ln_up1 = nn.LayerNorm(14 * 14)
        self.mha_up1 = ViMHA(16, 16, num_heads, patch_size)
        self.up2 = UpSampleBlock(16, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be of shape (B, 1, 28, 28) for MNIST-like images.
        """
        x = self.down1(x) # (B, 16, 14, 14)
        B, C, H, W = x.shape
        x = x.view(x.size(0) * x.size(1), -1)
        x = self.ln_down1(x)
        x = x.view(B, C, H, W)  # Reshape back to (B, 16, 14, 14)
        x = x + self.mha_down1(x)
        x = self.down2(x)
        B, C, H, W = x.shape
        x = x.view(x.size(0) * x.size(1), -1)
        x = self.ln_down2(x)
        x = x.view(B, C, H, W)
        x = x + self.mha_down2(x)
        x = self.up1(x)
        B, C, H, W = x.shape
        x = x.view(x.size(0) * x.size(1), -1)
        x = self.ln_up1(x)
        x = x.view(B, C, H, W)  
        x = x + self.mha_up1(x)
        x = self.up2(x)
        return x


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