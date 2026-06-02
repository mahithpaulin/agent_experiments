import numpy as np

class KernelRegression:
    def __init__(self, kernel='rbf', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.X_train = None
        self.y_train = None

    def _rbf_kernel(self, X1, X2):
        # Gaussian RBF kernel
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-dists / (2 * self.bandwidth ** 2))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model not trained yet")
        if self.kernel == 'rbf':
            K = self._rbf_kernel(X, self.X_train)
            weights = K / (np.sum(K, axis=1, keepdims=True) + 1e-12)
            return weights.dot(self.y_train)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

def special_training_trait(model, epoch):
    # Special training trait: adapt bandwidth to improve convergence
    # Decrease bandwidth exponentially to focus on local structure over epochs
    decay_rate = 0.9
    model.bandwidth *= decay_rate ** epoch
    # Clamp bandwidth to avoid too small values
    model.bandwidth = max(model.bandwidth, 0.05)