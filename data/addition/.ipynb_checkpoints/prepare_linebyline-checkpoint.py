"""
Prepare the Addition dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Get all the unique characters that occur in this text. Add <bos> and <eos> tokens to the character set
special_tokens = ['<bos>', '<eos>']
chars = sorted(list(set(data))) + special_tokens
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi['<bos>']] + [stoi[c] for c in s] + [stoi['<eos>']]  # encoder: add <bos> and <eos>, take a string, output a list of integers

def decode(l):
    return ''.join([itos[i] for i in l if i not in [stoi['<bos>'], stoi['<eos>']]])  # decoder: ignore <bos> and <eos>, take a list of integers, output a string


# Encode the data line by line
lines = data.strip().split(';')
lines = [s.strip('\n') for s in lines]  # Remove '\n' from each string in the list
encoded_data = [encode(line + ';') for line in lines[:-1]]

# Split data into train and validation sets
# np.random.shuffle(encoded_data)
split_idx = int(len(encoded_data) * 0.9)
train_ids = encoded_data[:split_idx]
val_ids = encoded_data[split_idx:]

train_sizes = [len(sublist) for sublist in train_ids]
val_sizes = [len(sublist) for sublist in val_ids]
print(f"train has {sum(train_sizes):,} tokens")
print(f"val has {sum(val_sizes):,} tokens")

# Save the encoded data
with open(os.path.join(os.path.dirname(__file__), 'train.pkl'), 'wb') as f:
    pickle.dump(train_ids, f)

with open(os.path.join(os.path.dirname(__file__), 'val.pkl'), 'wb') as f:
    pickle.dump(val_ids, f)

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

