import numpy as np

def generate_data(num_samples, noise=0.0):
    np.random.seed(42)
    X = np.random.rand(num_samples, 1) * 10  # Random values between 0 and 10
    true_weights = 2.5
    true_bias = 5.0
    y = true_weights * X.squeeze() + true_bias + np.random.normal(0, noise, num_samples)
    return X, y.reshape(-1, 1)