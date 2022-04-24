import random
import numpy as np
from PIL import Image
import scipy
from sklearn.decomposition import PCA
import torch

# from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import pickle as pkl

class FlickrDataset(Dataset):
    def __init__(self,image_embs, text_embs, aggregation, train, text=False, all_sent=False):
        self.train = train
        self.text = text
        self.all_sent = all_sent
        with open(image_embs, 'rb') as op:
            self.train_data = pkl.load(op)

        with open(text_embs, 'rb') as op:
            self.text_data = pkl.load(op)
        
        self.img_texts = []
        for (i,dat) in enumerate(self.train_data):
            self.train_data[i] = dat
            sentences = []
            for sent in self.text_data[i]:
                sentences.append(aggregation(sent))
            self.img_texts.append(sentences)
        # self.train_data = torch.stack(self.train_data)

        self.train_data_len = len(self.train_data)
        self.text_data_len = len(self.train_data) * len(self.img_texts[0])
        self.caps_per_image = len(self.img_texts[0])
        print(f'Img shape: {np.shape(self.train_data)}, Text shape: {np.shape(self.text_data)}, Captions: {self.text_data_len}')
        
    def __getitem__(self, index, rand = True): #ADDED RANDOM SETTING
        if self.all_sent:
            img, text = self.train_data[index], np.array(self.img_texts[index])
        elif rand and not self.text:
            img, text = self.train_data[index], self.img_texts[index][random.randint(0, len(self.img_texts[index])-1)]
        elif rand and self.text: #
            image_ind = index // self.caps_per_image
            text_ind = index % self.caps_per_image
            img, text = self.train_data[image_ind], self.img_texts[image_ind][text_ind]
        else:
            fused_sentences = self.img_texts[index][0]
            i = 1
            while i < len(self.img_texts[index]):
                fused_sentences += self.img_texts[index][i]
                i += 1
            img, text = self.train_data[index], fused_sentences/len(self.img_texts[index])

        return (img,text)
    
    def getAllCaptions(self, index):
        img, texts = self.train_data[index], self.img_texts[index]

        return (img,texts)

    def __len__(self):
        if self.text:
            return self.text_data_len 
        else:
            return self.train_data_len

#########
class FlickrDataset2(Dataset):
    def __init__(self,image_embs, text_embs, train):
        self.train = train

        with open(image_embs, 'rb') as op:
            self.train_data = pkl.load(op)

        with open(text_embs, 'rb') as op:
            self.text_data = pkl.load(op)
        
        self.train_data_len = len(self.train_data)
        
    def __getitem__(self, index):
        img, text = self.train_data[index], self.text_data[index][random.randint(0, len(self.text_data[index])-1)]

        return (img,text)
    
    def getAllCaptions(self, index):
        img, texts = self.train_data[index], self.text_data[index] #REVISE

        return (img,texts)

    def __len__(self):
        return self.train_data_len

#########

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
                if self.flickr_dataset.all_sent:
                    self.triplets.append((img, np.array(self.flickr_dataset.img_texts[ind]), np.array(self.flickr_dataset.img_texts[neg_indx])))
                else:
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
            neg_indx = index + 1

        return (img1, text_pos, text_neg), (index, index, neg_indx)

    def __len__(self):
        return len(self.flickr_dataset)

#########
class TripletFlickrDataset2(Dataset):
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
                text_pos = np.random.choice(range(len(self.flickr_dataset.text_data[ind])))
                neg_indx = ind
                while neg_indx == ind:
                    neg_indx = np.random.choice(range(len(self.flickr_dataset)))
                text_neg = np.random.choice(range(len(self.flickr_dataset.text_data[neg_indx])))
                self.triplets.append((img, self.flickr_dataset.text_data[ind][text_pos], self.flickr_dataset.text_data[neg_indx][text_neg]))

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
#########

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
