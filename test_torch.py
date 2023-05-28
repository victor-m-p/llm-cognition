import torch

device = torch.device('cuda:0')

# Create two random matrices
a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

# Multiply them
c = a @ b