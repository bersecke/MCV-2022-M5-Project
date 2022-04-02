import torch
from torch import embedding, nn
import torch.nn.functional as F

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
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 8),
        #     # nn.Softmax() #Softmax already included in the considered loss function
        # )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        embeddings = torch.flatten(x, 1)
        return embeddings

    def get_embedding(self, x):
        return self.forward(x)

    
class ClassificationNet(ModelM3):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
            # nn.LogSoftmax()
        )

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.classifier(output)
        return output

    def get_embedding(self, x):
        return self.embedding_net(x)

class ClassificationNet_v2(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet_v2, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.convnet = nn.Sequential(
                                    nn.Conv2d(3, 32, 5), nn.PReLU(),
                                    nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(32, 64, 5), nn.PReLU(),
                                    nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(238144, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
                                    nn.Conv2d(3, 32, 5), nn.PReLU(),
                                    nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(32, 64, 5), nn.PReLU(),
                                    nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(238144, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2),
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNet_V2(nn.Module):
    def __init__(self):
        super(EmbeddingNet_V2, self).__init__()
        self.convnet = nn.Sequential(
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
        self.fc = nn.Sequential(nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2),
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)