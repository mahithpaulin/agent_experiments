import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Import the components to be tested
from model import FastLinearNet
from main import generate_synthetic_data


@pytest.fixture(scope="module")
def synthetic_data():
    """Create a small synthetic dataset for testing."""
    num_samples = 128
    seq_len = 16
    input_dim = 32
    X, y = generate_synthetic_data(num_samples, seq_len, input_dim)
    return X, y


def test_generate_synthetic_data_shape(synthetic_data):
    """Ensure the synthetic data generator returns tensors of expected shapes."""
    X, y = synthetic_data
    assert X.dim() == 3  # (num_samples, seq_len, input_dim)
    assert y.dim() == 2  # (num_samples, 1)
    assert X.shape[0] == y.shape[0]  # same number of samples
    # Verify that the target equals the sum of all input elements
    expected = X.view(X.shape[0], -1).sum(dim=1, keepdim=True)
    torch.testing.assert_allclose(y, expected, atol=1e-6)


def test_fastlinearnet_forward_output():
    """Check that FastLinearNet produces an output of the correct shape."""
    batch_size = 8
    seq_len = 20
    input_dim = 24
    hidden_dim = 48
    output_dim = 1

    model = FastLinearNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
    )
    model.eval()

    # Random input tensor
    x = torch.randn(batch_size, seq_len, input_dim)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, output_dim)


def test_training_loss_decreases():
    """Run a tiny training loop and verify that the loss drops after a few steps."""
    torch.manual_seed(0)

    # Hyper‑parameters for a tiny dataset
    num_samples = 256
    seq_len = 12
    input_dim = 16
    hidden_dim = 32
    output_dim = 1
    batch_size = 32
    learning_rate = 1e-3
    num_steps = 5

    # Data
    X, y = generate_synthetic_data(num_samples, seq_len, input_dim)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = FastLinearNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Record initial loss on the first batch
    model.train()
    first_batch_X, first_batch_y = next(iter(loader))
    with torch.no_grad():
        init_pred = model(first_batch_X)
        init_loss = criterion(init_pred, first_batch_y).item()

    # Perform a few training steps
    for step, (batch_X, batch_y) in enumerate(loader):
        if step >= num_steps:
            break
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

    # Record loss after training steps on the same first batch
    with torch.no_grad():
        final_pred = model(first_batch_X)
        final_loss = criterion(final_pred, first_batch_y).item()

    # The loss should have decreased (allow a small tolerance for stochasticity)
    assert final_loss <= init_loss + 1e-4, f"Loss did not decrease: {init_loss:.6f} -> {final_loss:.6f}"