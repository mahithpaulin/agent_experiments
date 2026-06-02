import numpy as np

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse