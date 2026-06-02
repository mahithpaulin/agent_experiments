import numpy as np
from model import NeuralNetwork
from trainer import Trainer
from utils import preprocess_data, evaluate_model
import matplotlib.pyplot as plt

def main():
    # Hardcoded demo data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR problem

    # Preprocess data
    X, y = preprocess_data(X, y)

    # Initialize model and trainer
    input_size = X.shape[1]
    hidden_size = 4
    output_size = y.shape[1]
    learning_rate = 0.1

    model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    trainer = Trainer(model, learning_rate)

    # Train the model
    epochs = 1000
    losses = trainer.train(X, y, epochs)

    # Evaluate the model
    accuracy = evaluate_model(model, X, y)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Plot loss curve
    plt.plot(range(epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

if __name__ == "__main__":
    main()