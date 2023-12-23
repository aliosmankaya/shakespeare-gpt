import torch

from data import decode, vocab_size
from network import GPT, GPTConfig
from parameters import *

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
)

gptconf = GPTConfig(**model_args)
m = GPT(gptconf)
m.to(device)
m.load_state_dict(torch.load(path))
m.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

with open("gen.txt", "w") as f:
    text = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
    f.write(text)
