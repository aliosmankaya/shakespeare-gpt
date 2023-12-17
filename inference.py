import torch

from data import decode
from parameters import device

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode)
