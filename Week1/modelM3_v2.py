import torch
from torch import nn

# Define model
class ModelM3(nn.Module): 
    def __init__(self):
        super(ModelM3, self).__init__()
        self.features = nn.Sequential(
            
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, 7, padding="same"),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.GroupNorm(1, 64), #Equivalent to layer normalization
            # nn.Conv2d(64, 128, 3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, padding="same"),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding="same"),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.GroupNorm(1, 128), #Equivalent to layer normalization
            # nn.Conv2d(128, 256, 3),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128, padding="same"),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 8),
            # nn.Softmax() #Softmax already included in the considered loss function
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x