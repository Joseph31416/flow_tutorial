import torch
import torch.nn as nn

cols = 512
layer_norm = nn.LayerNorm(cols)

x = torch.rand(32, cols)
var_x_slice = torch.var(x[0, :])
exp_x_slice = torch.mean(x[0, :])
print(f"Variance of x[0, :]: {var_x_slice.item()}")
print(f"Expected value of x[0, :]: {exp_x_slice.item()}")

x_normed = layer_norm(x)
var_x_normed_slice = torch.var(x_normed[0, :])
exp_x_normed_slice = torch.mean(x_normed[0, :])
print(f"Variance of x_normed[0, :]: {var_x_normed_slice.item()}")
print(f"Expected value of x_normed[0, :]: {exp_x_normed_slice.item()}")
