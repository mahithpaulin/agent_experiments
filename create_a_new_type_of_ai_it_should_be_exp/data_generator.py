import numpy as np

def generate_data(samples=100, input_dim=1):
    X = np.random.rand(samples, input_dim)
    true_weights = np.random.rand(input_dim)
    true_bias = np.random.rand()
    y = np.dot(X, true_weights) + true_bias + np.random.normal(scale=0.1, size=samples)
    return X, y