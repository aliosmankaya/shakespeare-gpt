# Shakespeare-GPT

Generate Infinite Shakespeare Plays with GPT


<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Shakespeare.jpg/240px-Shakespeare.jpg'>

## Purpose

The purpose of this project is to use Self-Attention mechanism and GPT-2 architecture to understand its capabilities. Its focusing on Shakespeare Plays to train a GPT model and then generating plays like him.

## Install

Clone the project to your local environment:

```
git clone https://github.com/aliosmankaya/shakespeare-gpt.git
```

Install the necessary dependencies:

```
pip install -r requirements.txt
```

## Train

You can easily train the model:

```
python3 train.py
```

If you want to configure the parameters, you can change on [parameters.py](parameters.py)

Train output:

```
number of parameters: 10.65M
num decayed parameter tensors: 26, with 10,740,096 parameters
num non-decayed parameter tensors: 13, with 4,992 parameters
using fused AdamW: True
step 0: train loss 4.2873, val loss 4.2822
step 250: train loss 2.4680, val loss 2.4754
step 500: train loss 1.9606, val loss 2.0486
step 750: train loss 1.5911, val loss 1.7709
step 1000: train loss 1.4172, val loss 1.6190
step 1250: train loss 1.3100, val loss 1.5418
step 1500: train loss 1.2445, val loss 1.5046
step 1750: train loss 1.1911, val loss 1.4800
step 2000: train loss 1.1552, val loss 1.4792
step 2250: train loss 1.1138, val loss 1.4668
step 2500: train loss 1.0787, val loss 1.4655
step 2750: train loss 1.0492, val loss 1.4719
step 3000: train loss 1.0103, val loss 1.4768
step 3250: train loss 0.9776, val loss 1.4965
step 3500: train loss 0.9464, val loss 1.5046
step 3750: train loss 0.9112, val loss 1.5127
step 4000: train loss 0.8789, val loss 1.5482
step 4250: train loss 0.8467, val loss 1.5434
step 4500: train loss 0.8130, val loss 1.5735
step 4750: train loss 0.7830, val loss 1.5929
step 4999: train loss 0.7555, val loss 1.6410
```

## Inference

To generate Shakespeare plays with trained model:

```
python3 inference.py
```

Check the example output on [gen.txt](gen.txt)

## Citation

This project based on Andrej Karpathy's [video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [codes](https://github.com/karpathy/nanoGPT).