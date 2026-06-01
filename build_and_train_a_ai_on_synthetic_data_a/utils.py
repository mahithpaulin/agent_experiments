import torch

def generate_synthetic_data(samples, input_dim, output_dim):
    # Generate random inputs
    X = torch.randn(samples, input_dim)
    # Create a random linear relationship with noise
    weights = torch.randn(input_dim, output_dim)
    bias = torch.randn(output_dim)
    y = X @ weights + bias + 0.1 * torch.randn(samples, output_dim)
    return X, y