import pytest
import torch
from f import KAN, MLP, BSplineActivation
from main import target_function, generate_data, train_model, evaluate_model


def test_target_function_values():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    y = target_function(x)
    expected = torch.sin(torch.pi * x[:, 0]) + (x[:, 0] ** 2) * x[:, 1]
    assert torch.allclose(y, expected)


def test_bspline_activation_shapes_and_output_range():
    bspline = BSplineActivation(degree=3, n_knots=10, knot_range=(-2, 2))
    x = torch.linspace(-2, 2, 50)
    coeffs = torch.randn(10)
    out = bspline(x, coeffs)
    assert out.shape == x.shape
    # Output should be finite
    assert torch.isfinite(out).all()


def test_kan_layer_output_shape_and_grad():
    batch = 5
    input_dim = 2
    output_dim = 3
    x = torch.randn(batch, input_dim)
    kan_layer = KAN(input_dim=input_dim, hidden_dim=output_dim, output_dim=1)
    out = kan_layer(x)
    assert out.shape == (batch, 1)
    out.sum().backward()  # check backward pass


def test_mlp_output_shape_and_grad():
    batch = 5
    input_dim = 2
    output_dim = 1
    x = torch.randn(batch, input_dim)
    mlp = MLP(input_dim=input_dim, hidden_dim=16, output_dim=output_dim)
    out = mlp(x)
    assert out.shape == (batch, output_dim)
    out.sum().backward()


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_kan_outperforms_mlp(seed):
    torch.manual_seed(seed)
    x_train, y_train = generate_data(1000, seed=seed)
    x_test, y_test = generate_data(300, seed=seed + 1)

    input_dim = 2
    hidden_dim = 16
    output_dim = 1
    n_knots = 10
    degree = 3
    knot_range = (-2, 2)

    kan = KAN(input_dim, hidden_dim, output_dim, n_knots, degree, knot_range)
    mlp = MLP(input_dim, hidden_dim, output_dim)

    kan = train_model(kan, x_train, y_train, epochs=300, lr=1e-3)
    mlp = train_model(mlp, x_train, y_train, epochs=300, lr=1e-3)

    mse_kan = evaluate_model(kan, x_test, y_test)
    mse_mlp = evaluate_model(mlp, x_test, y_test)

    improvement = (mse_mlp - mse_kan) / mse_mlp
    assert improvement >= 0.10, f"KAN improvement {improvement:.3f} less than 10%"


def test_bspline_basis_sum_to_one():
    bspline = BSplineActivation(degree=3, n_knots=10, knot_range=(-2, 2))
    x = torch.linspace(-2, 2, 100)
    basis = bspline.bspline_basis(x)
    # Sum of basis functions at each x should be close to 1
    sums = basis.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_plot_spline_activations_runs_without_error():
    # Just test that plotting function runs without error
    from main import plot_spline_activations
    kan = KAN()
    x_train, y_train = generate_data(100)
    kan = train_model(kan, x_train, y_train, epochs=10, lr=1e-3)
    plot_spline_activations(kan.kan1)
    # If no exceptions, pass


# Run tests with pytest
# pytest test_main.py -v