import torch

from data import decode
from network import GPTLanguageModel
from parameters import device, path

m = GPTLanguageModel()
m.to(device)
m.load_state_dict(torch.load(path))
m.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
