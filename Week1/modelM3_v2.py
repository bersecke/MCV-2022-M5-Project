import torch
from torch import nn

# Define model
class ModelM3(nn.Module):
    def __init__(self):
        super(ModelM3, self).__init__()
        self.features = nn.Sequential(
            
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, 7),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            # nn.BatchNorm2d(64),
            # channel size = 122x122
            nn.LayerNorm([64, 61, 61]),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            # nn.BatchNorm2d(128),
            nn.LayerNorm([128, 29, 29]),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 8),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x