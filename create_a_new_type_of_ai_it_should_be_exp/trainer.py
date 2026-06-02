import numpy as np

class Trainer:
    def __init__(self, model, learning_rate=0.01, epochs=100):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        for epoch in range(self.epochs):
            grad_weights, grad_bias = self.model.compute_gradients(X, y)
            self.model.update_parameters(grad_weights, grad_bias, self.learning_rate)