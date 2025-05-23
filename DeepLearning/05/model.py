import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, activation="relu"):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = activation

    def forward(self, x):
        x = self.pool(self._act(self.conv1(x)))
        x = self.pool(self._act(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self._act(self.fc1(x))
        x = self.fc2(x)
        return x

    def _act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        else:
            raise ValueError("Unknown activation function")
