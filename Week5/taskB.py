import argparse
import numpy as np
import torch
from torch import nn
import pickle as pkl
from losses import TripletLoss
import torch

from torch.optim import lr_scheduler
import torch.optim as optim

import os
import numpy as np

from trainer import fit
cuda = torch.cuda.is_available()

from networks import EmbeddingNet, TripletNetAdapted, TripletNetAdaptedText

def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the inference script')
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='Initial LR')
    parser.add_argument('-d', '--dimension', default=128, type=int, help='New embedding space dimension')
    parser.add_argument('-m', '--margin', default=0.1, type=float, help='Triplet loss margin')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-n', '--normalize', default=False, type=bool, help='Perform L2 normalization to text embeddings')
    parser.add_argument('-id', '--img_dim', default=4096, type=int, help='Dimensions of image embeddings')
    parser.add_argument('-td', '--txt_dim', default=300, type=int, help='Dimension of text embeddings')
    parser.add_argument('-i', '--img_pth', default='./dataset/{}_img_embs.pkl', type=str, help='Path to image embeddings')
    parser.add_argument('-t', '--txt_pth', default='./dataset/{}_text_embs.pkl', type=str, help='Path to text embeddings')
    parser.add_argument('-o', '--output', default='trainedModels/trainedModel.pth', type=str, help='Output path to save the trained model')

    return parser.parse_args()

args = parse_args()
print('Training configuration: ',args)

cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

def aggregation_text(word_embs, axis = 0):
    return np.mean(word_embs, axis=axis)

def aggregation_text_concatenate(word_embs, words=20):
    base = np.zeros(words * len(word_embs[0]), dtype=np.float32)
    for ind, word in enumerate(word_embs[:20]):
        base[ind*words:(ind*words+len(word_embs[0]))] = word
    return base


# def aggregation_text(word_embs, axis = 0):
#     return np.mean(word_embs, axis=axis)

from flickrDataSet import *

# Prepare the dataset
# train_data = FlickrDataset('./dataset/train_img_embs.pkl', './dataset/bertTrain_text_embs.pkl', aggregation=aggregation_text, train= True)
# test_data = FlickrDataset('./dataset/test_img_embs.pkl', './dataset/bertTest_text_embs.pkl', aggregation=aggregation_text, train= False)
train_data = FlickrDataset(args.img_pth.format('train'), args.txt_pth.format('train'), aggregation=aggregation_text, train= True, text=False)
test_data = FlickrDataset(args.img_pth.format('test'), args.txt_pth.format('test'), aggregation=aggregation_text, train= False)

triplet_train_dataset = TripletFlickrDatasetText(train_data) # Returns triplet of images and target same/different
triplet_test_dataset = TripletFlickrDatasetText(test_data)

batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
img_emb_dim = args.img_dim
text_emb_dim = args.txt_dim

save_path = args.output

embedding_net_img = EmbeddingNet(emd_dim=img_emb_dim, out_dim=args.dimension, simple=True, activation=nn.ReLU())
embedding_net_text = EmbeddingNet(emd_dim=text_emb_dim, out_dim=args.dimension, simple=True, activation=nn.ReLU())
model = TripletNetAdaptedText(embedding_net_img, embedding_net_text)


model.to(device)

if not os.path.exists(save_path):
    margin = args.margin
    loss_fn = nn.TripletMarginLoss(margin)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.8, last_epoch=-1)
    n_epochs = args.epochs
    log_interval = 500
    ## Training !!!
    print('Starting training...!!')
    # try:
    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    # except:
    #     print('Error during training!!!')
    #     exit(-1)
    torch.save(model.state_dict(), save_path)
else:
    print('Loading model...')
    model.load_state_dict(torch.load(save_path))
    model.to(device)


