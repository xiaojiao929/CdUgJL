import random
import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    def __call__(self, sample):
        sample['image'] = torch.from_numpy(sample['image'].copy()).float()
        sample['seg'] = torch.from_numpy(sample['seg'].copy()).long()
        sample['quant'] = torch.from_numpy(sample['quant'].copy()).float()
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = sample['image'][:, :, ::-1].copy()
            seg = sample['seg']
            if seg.min() >= 0:
                sample['seg'] = seg[:, ::-1].copy()
        return sample


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = sample['image'][:, ::-1, :].copy()
            seg = sample['seg']
            if seg.min() >= 0:
                sample['seg'] = seg[::-1, :].copy()
        return sample


class RandomRotation90:
    def __call__(self, sample):
        k = random.randint(0, 3)
        if k > 0:
            sample['image'] = np.rot90(sample['image'], k, axes=(1, 2)).copy()
            seg = sample['seg']
            if seg.min() >= 0:
                sample['seg'] = np.rot90(seg, k, axes=(0, 1)).copy()
        return sample


class RandomIntensityJitter:
    def __init__(self, brightness=0.1, contrast=0.1):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        img = sample['image']
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = random.uniform(-self.brightness, self.brightness)
        sample['image'] = np.clip(img * alpha + beta, 0, 1)
        return sample


class Resize:
    def __init__(self, size):
        self.size = size  # (H, W)

    def __call__(self, sample):
        img = torch.from_numpy(sample['image']).float().unsqueeze(0)  # [1, C, H, W]
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=True).squeeze(0)
        sample['image'] = img.numpy()

        seg = sample['seg']
        if seg.min() >= 0:
            seg_t = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0)
            seg_t = F.interpolate(seg_t, size=self.size, mode='nearest').squeeze().long()
            sample['seg'] = seg_t.numpy()
        return sample


def get_transforms(input_size=(256, 256)):
    train_transforms = Compose([
        Resize(input_size),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        RandomRotation90(),
        RandomIntensityJitter(0.1, 0.1),
        ToTensor(),
    ])
    val_transforms = Compose([
        Resize(input_size),
        ToTensor(),
    ])
    return train_transforms, val_transforms
