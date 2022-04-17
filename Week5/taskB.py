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

from networks import EmbeddingNet, TripletNetAdaptedText

cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"


def aggregation_text(word_embs, axis = 0):
    return np.sum(word_embs, axis=axis)

from flickrDataSet import *

# Prepare the dataset
train_data = FlickrDataset('./dataset/train_img_embs.pkl', './dataset/train_text_embs.pkl', aggregation=aggregation_text, train= True)
test_data = FlickrDataset('./dataset/test_img_embs.pkl', './dataset/test_text_embs.pkl', aggregation=aggregation_text, train= False)

triplet_train_dataset = TripletFlickrDatasetText(train_data) # Returns triplet of images and target same/different
triplet_test_dataset = TripletFlickrDatasetText(test_data)

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
img_emb_dim = 4096
text_emb_dim = 300

save_path = 'trainedModels/TripletNet_taskB_01margin_50ep.pth'

embedding_net_img = EmbeddingNet(emd_dim=img_emb_dim, simple=True)
embedding_net_text = EmbeddingNet(emd_dim=text_emb_dim, simple=True)
model = TripletNetAdaptedText(embedding_net_img, embedding_net_text)


model.to(device)

if not os.path.exists(save_path):
    margin = 0.2
    loss_fn = nn.TripletMarginLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.9, last_epoch=-1)
    n_epochs = 30
    log_interval = 500
    ## Training !!!
    print('Starting training...!!')
    try:
        fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, wandb_plot=False)
    except:
        exit(-1)
    torch.save(model.state_dict(), save_path)
else:
    print('Loading model...')
    model.load_state_dict(torch.load(save_path))
    model.to(device)


