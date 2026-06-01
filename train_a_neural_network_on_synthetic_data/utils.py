import numpy as np
import torch

def generate_data(batch_size):
    x = np.random.randint(-100, 100, size=(batch_size, 2))
    # Randomly choose addition or subtraction for each pair
    ops = np.random.choice([0, 1], size=batch_size)  # 0:add, 1:sub
    y = np.where(ops == 0, x[:, 0] + x[:, 1], x[:, 0] - x[:, 1])
    return x.astype(np.float32), y.astype(np.float32)

def prepare_batch(x, y):
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    return x_tensor, y_tensor