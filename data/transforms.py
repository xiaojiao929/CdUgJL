#Amber
# MIT License
# Copyright (c) 2025 Amber Xiao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

class RandomFlip:
    """Randomly flip the image and mask horizontally."""
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        return {'image': image, 'mask': mask}

class Normalize:
    """Normalize image with given mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = (image - self.mean) / self.std
        return {'image': image, 'mask': mask}

class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        return {'image': image, 'mask': mask}

class Compose:
    """Compose several transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

def get_transforms(mode='train'):
    """Return composed transforms for different training modes."""
    if mode == 'train':
        return Compose([
            RandomFlip(),
            Normalize(mean=0.5, std=0.25),
            ToTensor()
        ])
    elif mode == 'val' or mode == 'test':
        return Compose([
            Normalize(mean=0.5, std=0.25),
            ToTensor()
        ])
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
