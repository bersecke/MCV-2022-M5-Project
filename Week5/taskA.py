import numpy as np
import torch
from torch import nn
import pickle as pkl
import torch
import wandb
# wandb.init(project='M5-Image-to-text', entity='fantastic5')

from torch.optim import lr_scheduler
import torch.optim as optim

import os
import numpy as np

from trainer import fit
cuda = torch.cuda.is_available()

import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

class EmbeddingNet(nn.Module):
    def __init__(self, emd_dim = 4096):
        super(EmbeddingNet, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(emd_dim, 256),
                                nn.PReLU(),
                                )

    def forward(self, x1):
        output1 = self.fc1(x1)
        return output1

    def get_embedding(self, x):
        return self.forward(x)

class TripletNetAdapted(nn.Module):
    def __init__(self, image_embedding_net, word_embedding_net):
        super(TripletNetAdapted, self).__init__()
        self.image_net = image_embedding_net
        self.text_net = word_embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.image_net(x1)
        output2 = self.text_net(x2)
        output3 = self.text_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

def aggregation_text(word_embs, axis = 0):
    return np.sum(word_embs, axis=axis)

from flickrDataSet import *

# Prepare the dataset
train_data = FlickrDataset('./dataset/train_img_embs.pkl', './dataset/train_text_embs.pkl', aggregation=aggregation_text, train= True)
test_data = FlickrDataset('./dataset/test_img_embs.pkl', './dataset/test_text_embs.pkl', aggregation=aggregation_text, train= False)

triplet_train_dataset = TripletFlickrDataset(train_data) # Returns triplet of images and target same/different
triplet_test_dataset = TripletFlickrDataset(test_data)

batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
img_emb_dim = 4096
text_emb_dim = 300

save_path = 'TripletNet.pth'

embedding_net_img = EmbeddingNet(emd_dim=img_emb_dim)
embedding_net_text = EmbeddingNet(emd_dim=text_emb_dim)
model = TripletNetAdapted(embedding_net_img, embedding_net_text)


model.to(device)

if not os.path.exists(save_path):
    margin = 0.5
    loss_fn = nn.TripletMarginLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.9, last_epoch=-1)
    n_epochs = 100
    log_interval = 10
    ## Training !!!
    print('Starting training...!!')
    try:
        fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,wandb_plot=False)
    except:
        exit(-1)
    torch.save(model.state_dict(), save_path)
else:
    print('Loading model...')
    model.load_state_dict(torch.load(save_path))
    model.to(device)


