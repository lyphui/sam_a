import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register


@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask
        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return self.img_process(x)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

    
import pandas as pd

@register('glue-image-list')
class ImageList(Dataset):
    def __init__(self, path, 
                 repeat=1, mask=False,suffix='.png'):
        self.repeat = repeat
        self.path = path
        self.mask = mask

        
        pd_data = pd.read_csv(path)
        image_path_list = list(pd_data['img_pth'])
        
        if self.mask:
            self.files = [i.replace('.jpg',suffix) for i in image_path_list]
        else:
            self.files=image_path_list
            
    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        return self.img_process(x)

    def img_process(self, file):
        if self.mask:
            img = Image.open(file).convert('L')
#             print('maskbf',file,img.size)
            if  img.size[0] < img.size[1]:  # rotate 90
                img = img.rotate(90,expand=True)
#             print('aft',img.size)
            return img
        else:
            img = Image.open(file).convert('RGB')
#             print('imgbf',img.size)
            if  img.size[0] < img.size[1]:  # rotate 90
                img = img.rotate(90,expand=True)
#             print('aft',img.size)
            return img
        
        
@register('paired-glue-image-list')
class PairedImageList(Dataset):

    def __init__(self, csv_path, **kwargs):
        self.dataset_1 = ImageList(csv_path, **kwargs)
        self.dataset_2 = ImageList(csv_path, **kwargs, mask=True)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
