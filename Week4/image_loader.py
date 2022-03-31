from PIL import Image
import pickle
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