import pytest
import torch
from main import SimpleNN, train_model, main
from utils import generate_synthetic_data

def test_generate_synthetic_data_shapes():
    samples, input_dim, output_dim = 50, 5, 2
    X, y = generate_synthetic_data(samples, input_dim, output_dim)
    assert X.shape == (samples, input_dim)
    assert y.shape == (samples, output_dim)

def test_train_model_decreasing_loss():
    torch.manual_seed(42)
    input_dim, hidden_dim, output_dim = 10, 15, 1
    samples = 200
    data, targets = generate_synthetic_data(samples, input_dim, output_dim)
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    losses = train_model(model, data, targets, epochs=25, lr=0.01)
    assert len(losses) == 25
    # Check losses are floats and positive
    for loss in losses:
        assert isinstance(loss, float)
        assert loss > 0
    # Assert loss decreases overall (final less than first)
    assert losses[-1] < losses[0]
    # Assert monotonic decrease with some tolerance (allow small increases)
    for i in range(1, len(losses)):
        assert losses[i] <= losses[i-1] + 1e-3

def test_main_runs_and_prints(capsys):
    main()
    captured = capsys.readouterr()
    lines = captured.out.strip().split('\n')
    assert len(lines) == 30
    # Each line should match expected format
    for i, line in enumerate(lines, 1):
        assert line.startswith(f"Epoch {i}: Loss=")
        # Extract loss float and check positive
        loss_str = line.split("Loss=")[1]
        loss_val = float(loss_str)
        assert loss_val > 0
    # Check decreasing trend in printed losses
    losses = [float(line.split("Loss=")[1]) for line in lines]
    assert losses[-1] < losses[0]
    for i in range(1, len(losses)):
        assert losses[i] <= losses[i-1] + 1e-3