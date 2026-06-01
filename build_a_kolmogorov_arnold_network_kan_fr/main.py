import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from f import KAN, MLP, BSplineActivation


def target_function(x):
    # x: (batch, 2)
    return torch.sin(torch.pi * x[:, 0]) + (x[:, 0] ** 2) * x[:, 1]


def generate_data(n_samples=1000, seed=42):
    torch.manual_seed(seed)
    x = (torch.rand(n_samples, 2) * 4) - 2  # Uniform in [-2, 2]^2
    y = target_function(x).unsqueeze(-1)
    return x, y


def train_model(model, x_train, y_train, epochs=500, lr=1e-3, verbose=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    return model


def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        mse = nn.functional.mse_loss(y_pred, y_test).item()
    return mse


def plot_spline_activations(kan_layer, knot_range=(-2, 2), n_points=200):
    # kan_layer: KANLayer instance
    # Plot learned spline activations per edge (output_dim x input_dim)
    degree = kan_layer.degree
    n_knots = kan_layer.n_knots
    knots = kan_layer.bspline.knots.detach().cpu()
    coeffs = kan_layer.coeffs.detach().cpu()  # (output_dim, input_dim, n_knots)

    x_vals = torch.linspace(knot_range[0], knot_range[1], n_points)
    bspline = BSplineActivation(degree=degree, n_knots=n_knots, knot_range=knot_range)
    bspline.knots.data = knots  # Use learned knots

    output_dim, input_dim, _ = coeffs.shape
    fig, axes = plt.subplots(output_dim, input_dim, figsize=(3 * input_dim, 3 * output_dim),
                             squeeze=False)
    for i in range(output_dim):
        for j in range(input_dim):
            c_ij = coeffs[i, j]
            y_vals = bspline(x_vals, c_ij)
            axes[i, j].plot(x_vals.numpy(), y_vals.numpy())
            axes[i, j].set_title(f"Output {i} Input {j}")
            axes[i, j].grid(True)
    plt.tight_layout()
    plt.savefig("spline_activations.png")
    plt.show()


def main():
    # Generate data
    x_train, y_train = generate_data(1000)
    x_test, y_test = generate_data(300, seed=123)

    # Instantiate models
    input_dim = 2
    hidden_dim = 16
    output_dim = 1
    n_knots = 10
    degree = 3
    knot_range = (-2, 2)

    kan = KAN(input_dim, hidden_dim, output_dim, n_knots, degree, knot_range)
    mlp = MLP(input_dim, hidden_dim, output_dim)

    # Train models
    kan = train_model(kan, x_train, y_train, epochs=500, lr=1e-3, verbose=True)
    mlp = train_model(mlp, x_train, y_train, epochs=500, lr=1e-3, verbose=True)

    # Evaluate models
    mse_kan = evaluate_model(kan, x_test, y_test)
    mse_mlp = evaluate_model(mlp, x_test, y_test)

    print(f"KAN MSE: {mse_kan:.6f}")
    print(f"MLP MSE: {mse_mlp:.6f}")

    # Check improvement at least 10%
    improvement = (mse_mlp - mse_kan) / mse_mlp
    print(f"KAN improvement over MLP: {improvement * 100:.2f}%")
    assert improvement >= 0.10, "KAN does not outperform MLP by at least 10% MSE"

    # Visualize learned spline activations per edge
    plot_spline_activations(kan.kan1, knot_range=knot_range)


if __name__ == "__main__":
    main()