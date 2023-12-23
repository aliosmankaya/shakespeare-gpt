import torch

eval_interval = 250
eval_iters = 200
log_interval = 10
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
warmup_iters = 100
weight_decay = 1e-1
bias = False
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
device = "cpu"
device_type = "cpu"
is_save = False
path = "./model.pt"
