import os
import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

CKPT = "tinycnn_weights.pt"

def get_model_and_input():
    torch.manual_seed(42)
    model = TinyCNN().eval()
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, weights_only=True))
    else:
        torch.save(model.state_dict(), CKPT)
    example_input = torch.randn(1, 3, 32, 32)
    return model, example_input