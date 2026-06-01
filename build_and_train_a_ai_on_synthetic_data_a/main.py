import torch
import torch.nn as nn
import torch.optim as optim
from utils import generate_synthetic_data

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, data, targets, epochs=20, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def main():
    torch.manual_seed(0)
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    samples = 1000

    data, targets = generate_synthetic_data(samples, input_dim, output_dim)
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    losses = train_model(model, data, targets, epochs=30, lr=0.01)

    # Print losses to show decreasing trend
    for i, loss in enumerate(losses):
        print(f"Epoch {i+1}: Loss={loss:.6f}")

if __name__ == "__main__":
    main()