import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_ch=1, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)   # → [B, n_patches, embed_dim]
        return x + self.pos_embed                     # add positional tokens

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=128, depth=6, n_heads=8, mlp_dim=256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(embed_dim, n_heads, mlp_dim, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.to_mu     = nn.Linear(embed_dim, embed_dim)
        self.to_logvar = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = x.transpose(0,1)                # [n_patches, B, embed_dim]
        x = self.encoder(x)
        x = x.mean(dim=0)                   # pool → [B, embed_dim]
        return self.to_mu(x), self.to_logvar(x)

class ViTDecoder(nn.Module):
    def __init__(self, img_size=28, patch_size=7, embed_dim=128, depth=6, n_heads=8, mlp_dim=256):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.z2patch = nn.Linear(embed_dim, n_patches * embed_dim)
        self.pos_embed= nn.Parameter(torch.randn(1, n_patches, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, n_heads, mlp_dim, activation='gelu')
        self.decoder  = nn.TransformerEncoder(layer, num_layers=depth)
        self.unproj   = nn.Linear(embed_dim, patch_size * patch_size)
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.n_patches  = n_patches
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # map z → sequence of patch embeddings
        x = self.z2patch(z).view(-1, self.n_patches, self.embed_dim)
        x = x + self.pos_embed
        x = self.decoder(x.transpose(0,1)).transpose(0,1)
        # project each patch back to pixels
        x = self.unproj(x)                              # [B, n_patches, patch²]
        B = x.size(0)
        x = x.view(B, self.n_patches, self.patch_size, self.patch_size)
        x = self.sigmoid(x)                          # apply sigmoid to get values between 0 and 1
        # reassemble into a full image
        return x.permute(0,1,2,3).reshape(B, 1, 28, 28)