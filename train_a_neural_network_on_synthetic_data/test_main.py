import pytest
import torch
import numpy as np
from main import MathNet, train
from utils import generate_data, prepare_batch

def test_generate_data_shapes_and_types():
    x, y = generate_data(10)
    assert x.shape == (10, 2)
    assert y.shape == (10,)
    assert x.dtype == y.dtype == np.float32

def test_prepare_batch_tensor_types_and_shapes():
    x_np, y_np = generate_data(5)
    x_tensor, y_tensor = prepare_batch(x_np, y_np)
    assert isinstance(x_tensor, torch.Tensor)
    assert isinstance(y_tensor, torch.Tensor)
    assert x_tensor.shape == (5, 2)
    assert y_tensor.shape == (5,)

def test_mathnet_forward_output_shape():
    model = MathNet()
    x_np, _ = generate_data(8)
    x_tensor = torch.from_numpy(x_np)
    output = model(x_tensor)
    assert output.shape == (8, 1)

def test_train_reduces_loss():
    model = MathNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    model.train()
    x_np, y_np = generate_data(64)
    x_tensor, y_tensor = prepare_batch(x_np, y_np)
    optimizer.zero_grad()
    output = model(x_tensor).squeeze()
    loss_before = criterion(output, y_tensor).item()
    loss_before_tensor = criterion(output, y_tensor)
    loss_before_tensor.backward()
    optimizer.step()
    optimizer.zero_grad()
    output_after = model(x_tensor).squeeze()
    loss_after = criterion(output_after, y_tensor).item()
    assert loss_after <= loss_before + 1e-4

@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_generate_data_values(batch_size):
    x, y = generate_data(batch_size)
    for i in range(batch_size):
        a, b = x[i]
        sum_val = a + b
        diff_val = a - b
        # y[i] should be either sum or diff
        assert y[i] == pytest.approx(sum_val) or y[i] == pytest.approx(diff_val)
