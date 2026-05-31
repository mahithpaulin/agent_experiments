import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import the FastLinearNet model defined in model.py
from model import FastLinearNet

def generate_synthetic_data(num_samples: int, seq_len: int, input_dim: int):
    """
    Generates a synthetic regression dataset.
    Each sample is a random sequence of shape (seq_len, input_dim).
    The target is the sum over the sequence and features, producing a scalar.
    """
    X = torch.randn(num_samples, seq_len, input_dim, dtype=torch.float32)
    # Target: sum of all elements per sample, reshaped to (num_samples, 1)
    y = X.view(num_samples, -1).sum(dim=1, keepdim=True)
    return X, y

def main():
    # Hyperparameters
    num_samples = 1024
    seq_len = 32
    input_dim = 64
    hidden_dim = 128
    output_dim = 1
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 10

    # Create synthetic dataset
    X, y = generate_synthetic_data(num_samples, seq_len, input_dim)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = FastLinearNet(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         seq_len=seq_len)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        model.train()

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= num_samples
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Time: {elapsed:.3f}s")

if __name__ == "__main__":
    main()