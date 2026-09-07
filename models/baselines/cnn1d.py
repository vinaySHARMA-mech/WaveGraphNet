# models/baselines/cnn1d.py
"""1D-CNN baseline: a stacked-convolution architecture over per-path sequences."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperCnnBaseline(nn.Module):
    """
    `in_channels` = `num_pairs * 2` for the raw input (amplitude+phase per
    path), or `1` for the ΔE-input fairness-ablation variant (see
    `data.datasets.Cnn1DDeltaEDataset`) -- no other code change needed.
    """

    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=2)
        self.pool_final = nn.AdaptiveAvgPool1d(1)
        self.fc_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = torch.flatten(self.pool_final(x), 1)
        return self.fc_head(x)
