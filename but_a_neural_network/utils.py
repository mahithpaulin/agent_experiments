import numpy as np

def preprocess_data(X, y):
    """
    Normalize input data and ensure labels are in the correct shape.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Labels.
    Returns:
        tuple: Normalized features and reshaped labels.
    """
    X_norm = X / np.max(X, axis=0)
    y_reshaped = y.reshape(-1, 1)
    return X_norm, y_reshaped

def evaluate_model(model, X, y):
    """
    Evaluate the model's accuracy.
    Args:
        model (NeuralNetwork): Trained model.
        X (np.ndarray): Input features.
        y (np.ndarray): True labels.
    Returns:
        float: Accuracy of the model.
    """
    predictions = model.forward(X)
    predictions = np.round(predictions)
    accuracy = np.mean(predictions == y)
    return accuracy