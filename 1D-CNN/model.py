import torch
import torch.nn as nn

class Anomaly1DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(Anomaly1DCNN, self).__init__()
        
        # Block 1 - large kernel to capture broad signal structure
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )


        
        # Regional Pooling - preserves 4 temporal bins to maintain shape topography
        self.global_pool = nn.AdaptiveAvgPool1d(4)
        
        # Classifier with hidden layer and dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
