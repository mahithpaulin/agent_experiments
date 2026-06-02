import pytest
import numpy as np
from model import LinearModel
from trainer import Trainer
from evaluator import Evaluator
from data_generator import generate_data

def test_generate_data():
    X, y = generate_data(samples=100, input_dim=1)
    assert X.shape == (100, 1)
    assert y.shape == (100,)

def test_model_initialization():
    model = LinearModel(input_dim=1)
    assert model.weights.shape == (1,)
    assert model.bias == 0

def test_model_prediction():
    model = LinearModel(input_dim=1)
    X = np.array([[1], [2], [3]])
    predictions = model.predict(X)
    assert predictions.shape == (3,)

def test_compute_gradients():
    model = LinearModel(input_dim=1)
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    grad_weights, grad_bias = model.compute_gradients(X, y)
    assert grad_weights.shape == (1,)
    assert isinstance(grad_bias, float)

def test_update_parameters():
    model = LinearModel(input_dim=1)
    grad_weights = np.array([0.1])
    grad_bias = 0.1
    model.update_parameters(grad_weights, grad_bias, learning_rate=0.01)
    assert model.weights[0] == -0.001
    assert model.bias == -0.001

def test_training():
    X, y = generate_data(samples=100, input_dim=1)
    model = LinearModel(input_dim=1)
    trainer = Trainer(model, learning_rate=0.01, epochs=10)
    trainer.train(X, y)
    assert model.weights.shape == (1,)

def test_evaluation():
    X, y = generate_data(samples=100, input_dim=1)
    model = LinearModel(input_dim=1)
    evaluator = Evaluator(model)
    mse = evaluator.evaluate(X, y)
    assert isinstance(mse, float)

def test_integration():
    X, y = generate_data(samples=100, input_dim=1)
    model = LinearModel(input_dim=1)
    trainer = Trainer(model, learning_rate=0.01, epochs=10)
    evaluator = Evaluator(model)
    trainer.train(X, y)
    mse = evaluator.evaluate(X, y)
    assert isinstance(mse, float)