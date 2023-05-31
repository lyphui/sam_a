import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode

import torch.nn.functional as F


def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset

        if isinstance(inp_size, int):
            inp_size = [inp_size, inp_size]

        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
            transforms.Resize(self.inp_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.inp_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        #         print('bf-----------', img.size, mask.size)
        #         print('aft-----------', self.img_transform(img).size, self.mask_transform(mask).size)
        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset

        if isinstance(inp_size, int):
            inp_size = [inp_size, inp_size]

        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
            transforms.Resize(self.inp_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1, 1, 1])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.inp_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # random filp
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        #         print('bf-----------', img.size, mask.size)
        #         print('aft-----------', self.img_transform(img).shape, self.mask_transform(mask).shape)
        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }

@register('val_padsquare')
class ValDataset_padsquare(Dataset):
    def __init__(self, dataset, inp_size=1056, augment=False):
        self.dataset = dataset

        if isinstance(inp_size, int):
            inp_size = [inp_size, inp_size]

        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
            transforms.Resize(self.inp_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.inp_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        w,h=  img.size
        assert w>=h,'shape not align w={} h={}'.format(w,h)
#         print('bf pad-----------', img.size, mask.size)
        pad = transforms.Pad((0, (w - h)//2))
#         pad = transforms.Pad((0, (w - h)//2), padding_mode='edge')
        img = pad(img)
        mask = pad(mask)
#         print('aft pad-----------', img.size, mask.size)

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }


@register('train_padsquare')
class TrainDataset_padsquare(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset

        if isinstance(inp_size, int):
            inp_size = [inp_size, inp_size]

        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
            transforms.Resize(self.inp_size),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=.2, contrast=.1, saturation=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1, 1, 1])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.inp_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        w, h = img.size
        assert w>=h,'shape not align w={} h={}'.format(w,h)
#         print('bf pad-----------', img.size, mask.size)
        pad = transforms.Pad((0, (w - h)//2))
#         pad = transforms.Pad((0, (w - h)//2), padding_mode='edge')
#         #img_pad = transforms.Pad((0, (w - h)//2), fill=(), )
#         #mask_pad = transforms.Pad((0, (w - h)//2))
        img = pad(img)
        mask = pad(mask)
#         print('aft pad-----------', img.size, mask.size)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }