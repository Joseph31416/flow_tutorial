import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timestep):
        device = timestep.device
        half_dim = self.embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / half_dim))
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super(UNet, self).__init__()

        self.pos_emb = SinusoidalPositionEmbeddings(embedding_dim)

        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.time_down1 = nn.Linear(embedding_dim, 128)
        self.time_down2 = nn.Linear(embedding_dim, 256)
        self.time_down3 = nn.Linear(embedding_dim, 512)
        self.time_down4 = nn.Linear(embedding_dim, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channels)

    def forward(self, x, t):
        t_embeddings = self.pos_emb(t)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = x2 + self.time_down1(t_embeddings).view(x2.size(0), -1, 1, 1)
        x3 = self.down2(x2)
        x3 = x3 + self.time_down2(t_embeddings).view(x3.size(0), -1, 1, 1)
        x4 = self.down3(x3)
        x4 = x4 + self.time_down3(t_embeddings).view(x4.size(0), -1, 1, 1)
        x5 = self.down4(x4)
        x5 = x5 + self.time_down4(t_embeddings).view(x5.size(0), -1, 1, 1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

if __name__ == "__main__":
    # Test positional embeddings
    # import numpy as np
    # embedding_dim = 64
    # pos_emb = SinusoidalPositionEmbeddings(embedding_dim)
    # sample_timestep = torch.tensor(np.arange(0, 1, 0.01, dtype=np.float32))
    # embeddings = pos_emb(sample_timestep)
    # print("Positional Embeddings Shape:", embeddings.shape)
    # print("Sample Embeddings:", embeddings[:5])  # Print first 5 embeddings for

    # Test UNet
    model = UNet(in_channels=1, out_channels=1, embedding_dim=64)
    sample_input = torch.randn(32, 1, 28, 28)
    sample_t = torch.tensor([0.5] * 32)  # Example timestep
    output = model(sample_input, sample_t)
    print("Output Shape:", output.shape)  # Should be (1, 1, 28, 28)

                 