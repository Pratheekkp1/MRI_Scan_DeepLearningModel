import torch
import torch.nn as nn
import torch.nn.functional as F


class MRI3DNet(nn.Module):
    """Small 3D CNN for whole-volume MRI classification.

    Three conv → ReLU → max-pool blocks followed by a two-layer classifier head.
    The head uses LazyLinear so the flattened size is inferred from the input
    volume on the first forward pass — run one dry-run batch before building
    the optimizer so the head's parameters are materialized and trained.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool3d(2)

        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
