import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class DeepFTSG_2_PathLoader(Dataset):
    def __init__(self, image_paths, bgSub_paths, flux_paths,  target_paths):

        self.image_paths = image_paths
        self.bgSub_paths = bgSub_paths
        self.flux_paths = flux_paths
        self.target_paths = target_paths

    def __getitem__(self, index):

        x = self.image_paths[index]
        x2 = self.bgSub_paths[index]
        x3 = self.flux_paths[index]
        y = self.target_paths[index]

        return x, x2, x3, y

    def __len__(self):

        return len(self.image_paths)


# load data from tuple
class DeepFTSG_2_TupleLoader(Dataset):
    def __init__(self, all_paths, img_size=(320,480)):
        self.img_size = img_size
        self.all_paths = all_paths

        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        ])

    def __getitem__(self, index):

        imgPath, bgSubPath, fluxPath, labelPath = self.all_paths[index]

        image = Image.open(imgPath)
        bgSub = Image.open(bgSubPath).convert('L')
        flux  = Image.open(fluxPath).convert('L')

        image2 = Image.merge('RGB', (bgSub, flux, flux))
        
        x = self.transforms(image)
        xNew = self.transforms(image2)

        x2 = np.zeros([3, self.img_size[0], self.img_size[1]])
        x2[0:2,:,:] = xNew[0:2,:,:]

        mask = cv2.imread(labelPath, 0)
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        mask_bin = ((mask > 170) * 255).astype('uint8')
        # dilate mask
        kernel = np.ones((3,3), np.uint8)
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=1) > 0
        
        y = torch.from_numpy(mask_bin).long()
        
        weights = torch.from_numpy((mask != 85) * (mask != 50)).float()

        return x, x2, y, weights

    def __len__(self):

        return len(self.all_paths)