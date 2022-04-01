from PIL import Image
import pickle
import torch
import glob
import random
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets



class MITSplitDataSet(Dataset):
    def __init__(self, image_path, labels_path, processor):
        """
        A dataset example where the class is embedded in the file names
        Args:
            image_path (string): path to a pickle file with one image path per line
            labels_path (string): path to a pickle file with one label in same order as image_path path per line
            processor (string): torch image processor
        """
        images_filenames = pickle.load(open(image_path,'rb'))
        self.images_filenames = ['..' + n[15:] for n in images_filenames]
        self.labels = pickle.load(open(labels_path,'rb'))

        self.img_processor = processor

        self.data_len = len(self.images_filenames)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.images_filenames[index]
        # Open image
        im_as_im = Image.open(single_image_path)

        img = self.img_processor(im_as_im)
        
        return (img, self.labels[index])

    def __len__(self):
        return self.data_len

class MITSplitDataset_v2(Dataset):
    def __init__(self,train_path,test_path, transform, train):
        # Transforms
        self.transform=transform
        # Read the csv file
        self.train=train
        if self.train:
            self.train_data_info = pd.read_csv(train_path,header=None)
            self.train_data =[] 
            
            print("printing train data length MIT Split")
            print(len(self.train_data_info.index))

            for (i,j) in enumerate(np.asarray(self.train_data_info.iloc[:, 1])):
                try : 
                    self.train_data.append(self.to_tensor(Image.open("../MIT_split/"+j))) 
                except : 
                    print(j)
            

            self.train_data = torch.stack(self.train_data)
            self.train_labels = np.asarray(self.train_data_info.iloc[:, 2])
            self.train_labels = torch.from_numpy(self.train_labels)

            self.train_data_len = len(self.train_data_info.index)

        else :
            self.test_data_info = pd.read_csv(test_path,header=None)
            self.test_data =[] 
            for (i,j) in enumerate(np.asarray(self.test_data_info.iloc[:, 1])):
                try : 
                    self.test_data.append(self.to_tensor(Image.open("../MIT_split/"+j))) 
                except : 
                    print(j)  

            self.test_data = torch.stack(self.test_data)
            self.test_labels = np.asarray(self.test_data_info.iloc[:, 2])
            self.test_labels = torch.from_numpy(self.test_labels)
            
            self.test_data_len = len(self.test_data_info.index)
            

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform:
            img = self.transform(img)

        return (img,target)

    def __len__(self):
        if self.train :
            return self.train_data_len
        else :
            return self.test_data_len


######## Work in progress
# class MITSplitDataset_v3(Dataset):
#     def __init__(self, train_data_path, test_data_path, train_labels_path, test_labels_path, transform, train):
#         # Transforms
#         self.transform=transform
#         # Read the csv file
#         self.train=train

#         if self.train:
#             train_image_paths = [] #to store image paths in list
#             classes = [] #to store class values

#             for data_path in glob.glob(train_data_path + '/*'):
#                 classes.append(data_path.split('/')[-1]) 
#                 train_image_paths.append(glob.glob(data_path + '/*'))

#             train_image_paths = list(flatten(train_image_paths))
#             random.shuffle(train_image_paths)

#             idx_to_class = {i:j for i, j in enumerate(classes)}
#             class_to_idx = {value:key for key,value in idx_to_class.items()}

#             self.train_images_filenames = train_image_paths
#             self.labels = pickle.load(open(train_labels_path,'rb'))
#             self.train_data =[] 
            
#             print("printing train data length CUHK")
#             print(len(self.train_data_info.index))

#             for img in train_images_filenames:
#                 try : 
#                     self.train_data.append(self.to_tensor(Image.open(img))) 
#                 except : 
#                     print(img)
            

#             self.train_data = torch.stack(self.train_data)
#             self.train_labels = pickle.load(open(train_labels_path,'rb'))

#             self.train_data_len = len(self.train_images_filenames)

#         else :
#             test_images_filenames = pickle.load(open(test_path,'rb'))
#             self.train_images_filenames = ['..' + n[15:] for n in test_images_filenames]
#             self.labels = pickle.load(open(train_labels_path,'rb'))
#             self.test_data =[] 
#             for img in test_images_filenames:
#                 try : 
#                     self.test_data.append(self.to_tensor(Image.open(img))) 
#                 except : 
#                     print(img)  

#             self.test_data = torch.stack(self.test_data)
#             self.test_labels = pickle.load(open(test_labels_path,'rb'))
            
#             self.test_data_len = len(self.test_images_filenames)
            

#     def __getitem__(self, index):
#         if self.train:
#             img, target = self.train_data[index], self.train_labels[index]
#         else:
#             img, target = self.test_data[index], self.test_labels[index]

#         if self.transform:
#             img = self.transform(img)

#         return (img,target)

#     def __len__(self):
#         if self.train :
#             return self.train_data_len
#         else :
#             return self.test_data_len