import numpy as np

class Trainer:
    def __init__(self, model, learning_rate=0.1):
        self.model = model
        self.learning_rate = learning_rate

    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            predictions = self.model.forward(X)

            # Compute loss (Mean Squared Error)
            loss = np.mean((y - predictions) ** 2)
            losses.append(loss)

            # Backward pass
            self.model.backward(X, y, predictions)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return losses