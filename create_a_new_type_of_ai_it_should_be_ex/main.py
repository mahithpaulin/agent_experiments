import numpy as np
from utils import KernelRegression, special_training_trait

class ExplainableAI:
    def __init__(self, kernel='rbf', bandwidth=1.0):
        self.model = KernelRegression(kernel=kernel, bandwidth=bandwidth)
        self.trained = False

    def fit(self, X, y, epochs=10):
        for epoch in range(epochs):
            self.model.fit(X, y)
            special_training_trait(self.model, epoch)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Model not trained yet")
        return self.model.predict(X)

if __name__ == "__main__":
    # Simple demo
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sinc(X).ravel()
    ai = ExplainableAI(bandwidth=0.5)
    ai.fit(X, y, epochs=5)
    preds = ai.predict(X)
    print("Predictions:", preds[:5])