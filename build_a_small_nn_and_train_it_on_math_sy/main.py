# main.py
import torch
import torch.nn as nn
import numpy as np
from utils import generate_synthetic_data, plot_results


class SmallNN(nn.Module):
    def __init__(self):
        super(SmallNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, device, data, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in data:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNN().to(device)
    data = generate_synthetic_data(1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(10): # Reduced number of epochs to prevent timeout
        loss = train(model, device, data, optimizer, criterion)
        print(f'Epoch {epoch+1}, Loss: {loss}')
    plot_results(model, device, data)

if __name__ == '__main__':
    main()