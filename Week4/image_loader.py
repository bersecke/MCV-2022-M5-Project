from PIL import Image
import pickle
import torch
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
    def __init__(self, train_path, test_path, train_labels_path, test_labels_path,train):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.transform=None
        # Read the csv file
        self.train=train
        if self.train:
            train_images_filenames = pickle.load(open(train_path,'rb'))
            self.train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
            self.labels = pickle.load(open(train_labels_path,'rb'))
            self.train_data =[] 
            
            print("printing train data length CUHK")
            print(len(self.train_data_info.index))

            for img in train_images_filenames:
                try : 
                    self.train_data.append(self.to_tensor(Image.open(img))) 
                except : 
                    print(img)
            

            self.train_data = torch.stack(self.train_data)
            self.train_labels = pickle.load(open(train_labels_path,'rb'))

            self.train_data_len = len(self.train_images_filenames)

        else :
            test_images_filenames = pickle.load(open(test_path,'rb'))
            self.train_images_filenames = ['..' + n[15:] for n in test_images_filenames]
            self.labels = pickle.load(open(train_labels_path,'rb'))
            self.test_data =[] 
            for img in test_images_filenames:
                try : 
                    self.test_data.append(self.to_tensor(Image.open(img))) 
                except : 
                    print(img)  

            self.test_data = torch.stack(self.test_data)
            self.test_labels = pickle.load(open(test_labels_path,'rb'))
            
            self.test_data_len = len(self.test_images_filenames)
            

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return (img,target)

    def __len__(self):
        if self.train :
            return self.train_data_len
        else :
            return self.test_data_len