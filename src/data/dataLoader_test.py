import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from PIL import Image


class DeepFTSG_1_TestDataLoader(Dataset):
    def __init__(self, image_paths, bgSub_paths, flux_paths, img_size=(320,480)):

        self.image_paths = image_paths
        self.bgSub_paths = bgSub_paths
        self.flux_paths = flux_paths
        self.img_size = img_size

        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        ])

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert('L')
        bgSub = Image.open(self.bgSub_paths[index]).convert('L')
        flux = Image.open(self.flux_paths[index]).convert('L')

        image2 = Image.merge('RGB', (image, bgSub, flux))

        x = self.transforms(image2)

        return x


    def __len__(self):

        return len(self.image_paths)


# test data loader for DeepFTSG_TwoStream
class DeepFTSG_2_TestDataLoader(Dataset):
    def __init__(self, image_paths, bgSub_paths, flux_paths, img_size=(320,480)):

        self.image_paths = image_paths
        self.bgSub_paths = bgSub_paths
        self.flux_paths = flux_paths
        self.img_size = img_size

        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        ])

    def __getitem__(self, index):
	
        image = Image.open(self.image_paths[index])
        # bgSub mask
        bgSub = Image.open(self.bgSub_paths[index]).convert('L')
        # flux mask
        flux = Image.open(self.flux_paths[index]).convert('L')
        
        image2 = Image.merge('RGB', (bgSub, flux, flux))

        x = self.transforms(image)
        xNew = self.transforms(image2)

        x2 = np.zeros([3, self.img_size[0], self.img_size[1]])
        x2[0:2,:,:] = xNew[0:2,:,:]

        return x, x2	
        

    def __len__(self):

        return len(self.image_paths)
        

class DeepFTSG_2_SBI_TestDataLoader(Dataset):
    def __init__(self, image_paths, bgSub_paths, flux_paths, img_size=(320,480)):

        self.image_paths = image_paths
        self.bgSub_paths = bgSub_paths
        self.flux_paths = flux_paths
        self.img_size = img_size

        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        ])

    def __getitem__(self, index):
	
        image = Image.open(self.image_paths[index]).convert('RGB')
        # bgSub mask
        bgSub = Image.open(self.bgSub_paths[index]).convert('L')
        # flux mask
        flux = Image.open(self.flux_paths[index]).convert('L')
        
        image2 = Image.merge('RGB', (bgSub, flux, flux))

        x = self.transforms(image)
        xNew = self.transforms(image2)

        x2 = np.zeros([3, self.img_size[0], self.img_size[1]])
        x2[0:2,:,:] = xNew[0:2,:,:]

        return x, x2	
        

    def __len__(self):

        return len(self.image_paths)