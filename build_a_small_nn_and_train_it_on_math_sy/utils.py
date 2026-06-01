# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_data(size):
    x = np.random.rand(size, 2)
    y = x[:, 0] + 2 * x[:, 1]
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float().view(-1, 1)
    return list(zip(x, y))


def plot_results(model, device, data):
    x, y = zip(*data)
    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)
    with torch.no_grad():
        predictions = model(x)
    plt.scatter(y.cpu().numpy(), predictions.cpu().numpy())
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.savefig('output.png')
    plt.show()