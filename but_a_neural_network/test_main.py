import pytest
import numpy as np
from model import NeuralNetwork
from trainer import Trainer
from utils import preprocess_data, evaluate_model

def test_preprocess_data():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    X_norm, y_reshaped = preprocess_data(X, y)

    assert np.allclose(X_norm, np.array([[0.33333333, 0.5], [1., 1.]]))
    assert y_reshaped.shape == (2, 1)

def test_neural_network():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = X.shape[1]
    hidden_size = 4
    output_size = y.shape[1]
    learning_rate = 0.1

    model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    trainer = Trainer(model, learning_rate)

    # Train the model
    epochs = 1000
    losses = trainer.train(X, y, epochs)

    # Check loss decreases
    assert losses[0] > losses[-1]

def test_evaluate_model():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = X.shape[1]
    hidden_size = 4
    output_size = y.shape[1]
    learning_rate = 0.1

    model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    trainer = Trainer(model, learning_rate)

    # Train the model
    epochs = 1000
    trainer.train(X, y, epochs)

    # Evaluate the model
    accuracy = evaluate_model(model, X, y)
    assert accuracy == pytest.approx(1.0, rel=1e-2)