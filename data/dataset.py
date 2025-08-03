# Amber
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

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from torchvision import transforms

class MedicalImageDataset(Dataset):
    """
    Dataset for loading paired non-contrast and contrast-enhanced MRI images along with masks and quantification labels.
    """
    def __init__(self, root_dir, modalities, transform=None, labeled=True):
        super(MedicalImageDataset, self).__init__()
        self.root_dir = root_dir
        self.modalities = modalities
        self.transform = transform
        self.labeled = labeled

        self.samples = self._load_samples()

    def _load_samples(self):
        sample_list = []
        for subject in os.listdir(self.root_dir):
            subj_path = os.path.join(self.root_dir, subject)
            if not os.path.isdir(subj_path):
                continue

            try:
                sample = {
                    "T2FS": os.path.join(subj_path, "T2FS.nii.gz"),
                    "DWI": os.path.join(subj_path, "DWI.nii.gz"),
                    "mask": os.path.join(subj_path, "mask.nii.gz") if self.labeled else None,
                    "quant": os.path.join(subj_path, "quant.npy") if self.labeled else None
                }
                sample_list.append(sample)
            except:
                continue
        return sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = []

        for mod in self.modalities:
            img = nib.load(sample[mod]).get_fdata()
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # CxHxW
            inputs.append(img)

        image = torch.cat(inputs, dim=0)

        if self.transform:
            image = self.transform(image)

        output = {"image": image}

        if self.labeled:
            mask = nib.load(sample["mask"]).get_fdata()
            mask = torch.tensor(mask, dtype=torch.long)
            quant = np.load(sample["quant"])
            output.update({
                "mask": mask,
                "quant": torch.tensor(quant, dtype=torch.float32)
            })

        return output
