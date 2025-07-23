import torch
import torch.nn as nn
from modules import DownBlockWithResAttn, UpBlockWithResAttn

class ViTUnetResNorm(nn.Module):

    def __init__(self,
                #  in_channels: int, out_channels: int,
                 channels: list[int],
                 num_heads: list[int],
                 patch_sizes: list[int],
                 init_h: int, init_w: int):
        super(ViTUnetResNorm, self).__init__()
        self.down_blocks = nn.ModuleList()
        in_channels = channels[0]
        for out_channels, num_head, patch_size in zip(channels[1:], num_heads, patch_sizes):
            self.down_blocks.append(
                DownBlockWithResAttn(in_channels, out_channels, num_head, patch_size, init_h, init_w)
            )
            in_channels = out_channels
            init_h //= 2
            init_w //= 2
        self.up_blocks = nn.ModuleList()
        for out_channels, num_head, patch_size in zip(
            reversed(channels[:-1]), reversed(num_heads[:-1]), reversed(patch_sizes[:-1])
        ):
            self.up_blocks.append(
                UpBlockWithResAttn(in_channels, out_channels, num_head, patch_size, init_h, init_w)
            )
            in_channels = out_channels
            init_h *= 2
            init_w *= 2
        self.up_blocks.append(
            UpBlockWithResAttn(in_channels, channels[0], num_heads[0], patch_sizes[0], init_h, init_w)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be of shape (B, 1, 28, 28) for MNIST-like images.
        """
        for down_block in self.down_blocks:
            x = down_block(x)
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x)
        return x
