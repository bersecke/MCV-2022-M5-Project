import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset  # For custom datasets
import torch
from torch.optim import lr_scheduler
import torch.optim as optim

from trainer import fit
import numpy as np
from datasets import SiameseMNIST

# %matplotlib inline
import matplotlib.pyplot as plt

mit_classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain',
                         'Opencountry', 'street', 'tallbuilding']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',]

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(8):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mit_classes)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

"""# SiameseNet"""

class MITSplitDataset_v2(Dataset):
    def __init__(self,train_path,test_path, transform, train):
        # Transforms
        self.transform=transform
        self.train=train
        self.to_tensor = transforms.ToTensor()

        # Read the csv file
        if self.train:
            self.train_data_info = pd.read_csv(train_path,header=None)
            self.train_data =[] 
            
            print("printing train data length MIT Split")
            # print(len(self.train_data_info.index))

            for (i,j) in enumerate(np.asarray(self.train_data_info.iloc[:, 1])):
                try :
                    self.train_data.append(self.to_tensor(Image.open(j))) 
                except : 
                    print('ERROR LOADING:' + j)
            
            self.train_data = torch.stack(self.train_data)
            self.train_labels = np.asarray(self.train_data_info.iloc[:, 2])
            self.train_labels = torch.from_numpy(self.train_labels)

            self.train_data_len = len(self.train_data_info.index)

        else :
            self.test_data_info = pd.read_csv(test_path,header=None)
            self.test_data =[] 
            for (i,j) in enumerate(np.asarray(self.test_data_info.iloc[:, 1])):
                try : 
                    self.test_data.append(self.to_tensor(Image.open(j))) 
                except : 
                    print('ERROR LOADING:' + j) 

            self.test_data = torch.stack(self.test_data)
            self.test_labels = np.asarray(self.test_data_info.iloc[:, 2])
            self.test_labels = torch.from_numpy(self.test_labels)
            
            self.test_data_len = len(self.test_data_info.index)
            

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # if self.transform:
        #     img = self.transform(img)

        return (img,target)

    def __len__(self):
        if self.train :
            return self.train_data_len
        else :
            return self.test_data_len

cuda = torch.cuda.is_available()

TRAIN_DATA_PATH = "../MIT_split/train_data.csv"
TEST_DATA_PATH = "../MIT_split/test_data.csv"

# from image_loader import MITSplitDataset_v2

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4275, 0.4525, 0.4494],
                         std=[0.2566, 0.2470, 0.2764])
    ])

TRANSFORM_IMG_TEST = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4275, 0.4525, 0.4494],
                         std=[0.2566, 0.2470, 0.2764])
    ])

train_data = MITSplitDataset_v2(TRAIN_DATA_PATH, TEST_DATA_PATH, transform=TRANSFORM_IMG, train= True)
test_data = MITSplitDataset_v2(TRAIN_DATA_PATH, TEST_DATA_PATH, transform=TRANSFORM_IMG_TEST, train= False)

siamese_train_dataset = SiameseMNIST(train_data) # Returns pairs of images and target same/different
siamese_test_dataset = SiameseMNIST(test_data)

batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss

# Step 2
embedding_net = EmbeddingNet()
# Step 3
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
    
# Step 4
margin = 1.
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.9, last_epoch=-1)
n_epochs = 20
log_interval = 500

fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)