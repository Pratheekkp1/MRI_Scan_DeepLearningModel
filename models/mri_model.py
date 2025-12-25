import torch
import torch.nn as nn
import torch.nn.functional as F


class MRI3DNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool3d(2)

        # Placeholder for dynamic flatten size
        self._flattened_size = None

        # Will initialize these lazily once we know volume size
        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes

    def _lazy_init_fc(self, x):
        # x is the tensor after conv layers and pooling
        flattened = torch.flatten(x, 1).shape[1]
        self._flattened_size = flattened

        # Now create the linear layers dynamically
        self.fc1 = nn.Linear(flattened, 128).to(x.device)
        self.fc2 = nn.Linear(128, self.num_classes).to(x.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Lazy layer creation
        if self.fc1 is None:
            self._lazy_init_fc(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
