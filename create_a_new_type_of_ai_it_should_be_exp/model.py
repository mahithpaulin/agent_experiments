import numpy as np

class LinearModel:
    def __init__(self, input_dim):
        self.weights = np.zeros((input_dim,))
        self.bias = 0

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def compute_gradients(self, X, y):
        predictions = self.predict(X)
        errors = predictions - y
        grad_weights = (2 / X.shape[0]) * np.dot(X.T, errors)
        grad_bias = (2 / X.shape[0]) * np.sum(errors)
        return grad_weights, grad_bias

    def update_parameters(self, grad_weights, grad_bias, learning_rate):
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias