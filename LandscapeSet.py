from torch.utils.data import Dataset
import cv2
import os


class LandscapeSet(Dataset):
    
    def __init__(self, 
                 data, 
                 directory, 
                 transform = None):
        self.data      = data
        self.directory = directory
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # import
        path  = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)
            
        # augmentations
        if self.transform is not None:
            image = self.transform(image = image)['image']
        
        return image