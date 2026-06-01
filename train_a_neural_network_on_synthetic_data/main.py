import torch
import torch.nn as nn
import torch.optim as optim
from utils import generate_data, prepare_batch

class MathNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train(model, optimizer, criterion, epochs=1000, batch_size=64):
    model.train()
    for epoch in range(epochs):
        x, y = generate_data(batch_size)
        x_tensor, y_tensor = prepare_batch(x, y)
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output.squeeze(), y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def main():
    model = MathNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train(model, optimizer, criterion)
    torch.save(model.state_dict(), "mathnet.pth")

if __name__ == "__main__":
    main()