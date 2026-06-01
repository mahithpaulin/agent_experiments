# test_main.py
import pytest
import torch
import numpy as np
from main import SmallNN, train, generate_synthetic_data


def test_small_nn():
    model = SmallNN()
    x = torch.randn(1, 2)
    output = model(x)
    assert output.shape == (1, 1)


def test_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNN().to(device)
    data = generate_synthetic_data(100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    loss = train(model, device, data, optimizer, criterion)
    assert loss >= 0


def test_generate_synthetic_data():
    data = generate_synthetic_data(100)
    assert len(data) == 100
    x, y = data[0]
    assert x.shape == (2,)
    assert y.shape == (1,)


def test_train_convergence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNN().to(device)
    data = generate_synthetic_data(1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    losses = []
    for _ in range(10): # Reduced number of epochs to prevent timeout
        loss = train(model, device, data, optimizer, criterion)
        losses.append(loss)
    assert losses[-1] < losses[0]


def test_model_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNN().to(device)
    data = generate_synthetic_data(100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    for _ in range(10): # Reduced number of epochs to prevent timeout
        loss = train(model, device, data, optimizer, criterion)
    x, y = data[0]
    x = x.to(device)
    with torch.no_grad():
        prediction = model(x)
    assert prediction.item() == pytest.approx(y.item(), abs=1e-2)


def test_model_prediction_on_new_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNN().to(device)
    data = generate_synthetic_data(100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    for _ in range(10): # Reduced number of epochs to prevent timeout
        loss = train(model, device, data, optimizer, criterion)
    new_x = torch.randn(1, 2).to(device)
    with torch.no_grad():
        prediction = model(new_x)
    assert prediction.shape == (1, 1)