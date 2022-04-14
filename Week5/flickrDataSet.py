import random
import numpy as np
from PIL import Image
import scipy
from sklearn.decomposition import PCA
import torch

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import pickle as pkl

class FlickrDataset(Dataset):
    def __init__(self,image_embs, text_embs, aggregation, train):
        self.train = train

        with open(image_embs, 'rb') as op:
            self.train_data = pkl.load(op)

        with open(text_embs, 'rb') as op:
            self.text_data = pkl.load(op)
            print(np.shape(self.text_data))
        
        self.img_texts = []
        for (i,dat) in enumerate(self.train_data):
            self.train_data[i] = dat
            sentences = []
            for sent in self.text_data[i]:
                sentences.append(aggregation(sent))
            self.img_texts.append(sentences)

        # self.train_data = torch.stack(self.train_data)

        self.train_data_len = len(self.train_data)
        

    def __getitem__(self, index):
        img, text = self.train_data[index], self.img_texts[index][random.randint(0, len(self.img_texts[index])-1)]

        return (img,text)
    
    def getAllCaptions(self, index):
        img, texts = self.train_data[index], self.img_texts[index]

        return (img,texts)

    def __len__(self):
        return self.train_data_len


class TripletFlickrDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, flickr_dataset):
        self.flickr_dataset = flickr_dataset
        self.train = self.flickr_dataset.train

        if not self.train:
            self.triplets = []
            for ind in range(len(self.flickr_dataset)):
                img = self.flickr_dataset.train_data[ind]
                text_pos = np.random.choice(range(len(self.flickr_dataset.img_texts[ind])))
                neg_indx = ind
                while neg_indx == ind:
                    neg_indx = np.random.choice(range(len(self.flickr_dataset)))
                text_neg = np.random.choice(range(len(self.flickr_dataset.img_texts[neg_indx])))
                self.triplets.append((img, self.flickr_dataset.img_texts[ind][text_pos], self.flickr_dataset.img_texts[neg_indx][text_neg]))

    def __getitem__(self, index):
        if self.train:
            img1, text_pos = self.flickr_dataset[index]
            neg_indx = index
            while neg_indx == index:
                neg_indx = np.random.choice(range(len(self.flickr_dataset)))
            img2, text_neg = self.flickr_dataset[neg_indx]

        else:
            img1 = self.triplets[index][0]
            text_pos = self.triplets[index][1]
            text_neg = self.triplets[index][2]

        return (img1, text_pos, text_neg), []

    def __len__(self):
        return len(self.flickr_dataset)



class TripletFlickrDatasetText(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, flickr_dataset):
        self.flickr_dataset = flickr_dataset
        self.train = self.flickr_dataset.train

        if not self.train:
            self.triplets = []
            for ind in range(len(self.flickr_dataset)):
                img = self.flickr_dataset.train_data[ind]
                text_pos = np.random.choice(range(len(self.flickr_dataset.img_texts[ind])))
                neg_indx = ind
                while neg_indx == ind:
                    neg_indx = np.random.choice(range(len(self.flickr_dataset)))
                self.triplets.append((self.flickr_dataset.img_texts[ind][text_pos], img, self.flickr_dataset.train_data[neg_indx]))

    def __getitem__(self, index):
        if self.train:
            img1, text_pos = self.flickr_dataset[index]
            neg_indx = index
            while neg_indx == index:
                neg_indx = np.random.choice(range(len(self.flickr_dataset)))
            img2, text_neg = self.flickr_dataset[neg_indx]

        else:
            text_pos = self.triplets[index][0]
            img1 = self.triplets[index][1]
            img2 = self.triplets[index][2]

        return (text_pos, img1, img2), []

    def __len__(self):
        return len(self.flickr_dataset)
