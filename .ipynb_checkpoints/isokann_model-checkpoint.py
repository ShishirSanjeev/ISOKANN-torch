# isokann_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ISOKANN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = MLP(input_dim, hidden_dim, output_dim)
        self.output_dim = output_dim

    def train(self, data, n_epochs=1000, lr=1e-3):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        x = torch.tensor(data[:, :-1, :], dtype=torch.float32).reshape(-1, data.shape[2])
        y = torch.tensor(data[:, 1:, :], dtype=torch.float32).reshape(-1, data.shape[2])

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            chi_x = self.model(x)
            chi_y = self.model(y)
            koopman = chi_y.detach()
            S_chi_x = shift_scale(chi_x)
            loss = loss_fn(S_chi_x, koopman)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

    def predict(self, data):
        if data.ndim == 3:
            x = torch.tensor(data[:, :-1, :], dtype=torch.float32).reshape(-1, data.shape[2])
        elif data.ndim == 2:
            x = torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError("Unexpected data dimension in predict")
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        # output shape: (batch * time, output_dim)
        # reshape back if 3D input
        if data.ndim == 3:
            output = output.reshape(data.shape[0], data.shape[1] - 1, -1)
        return output.numpy()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

def shift_scale(x):
    x_centered = x - x.mean(dim=0, keepdim=True)
    x_scaled = x_centered / (x_centered.norm(dim=0, keepdim=True) + 1e-8)
    return x_scaled
