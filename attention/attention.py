import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

B, T, C = 4, 8, 32
head_size = 16

x = torch.randn(B, T, C)

proj_q = torch.rand(C, head_size, requires_grad=True)
proj_k = torch.rand(C, head_size, requires_grad=True)

q = x @ proj_q
print(f"Query Tensor")
print(q.shape)

k = x @ proj_k
print(f"Key Tensor")
print(k.shape)

qk = q @ k.transpose(-2, -1) * (C ** -0.5)
print(f"QK Tensor")
print(qk.shape)

qk_masked = torch.tril(qk)
print(f"Lower Tri QK Tensor")
print(qk_masked)
qk_masked = qk_masked.masked_fill(qk_masked == 0, float('-inf'))
print(f"Masked Lower Tri QK Tensor")
print(qk_masked)
attn_scores = F.softmax(qk_masked, dim=-1)
print(f"Attention Scores")
print(attn_scores)

