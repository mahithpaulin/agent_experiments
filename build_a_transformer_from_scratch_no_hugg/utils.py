import random
import torch

def generate_synthetic_data(token_to_idx, seq_len=16, num_samples=1000):
    vocab_size = len(token_to_idx)
    data = []
    for _ in range(num_samples):
        seq = [random.randint(0, vocab_size-1) for _ in range(seq_len)]
        input_seq = seq[:-1]
        target_seq = seq[1:]
        data.append((input_seq, target_seq))
    return data

def idx_to_text(indices, idx_to_token):
    return ''.join(idx_to_token[i] for i in indices)