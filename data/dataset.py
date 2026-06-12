import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MedicalImageDataset(Dataset):
    """
    Multi-modal MRI dataset for liver tumor segmentation and quantification.

    Expected directory layout:
        root/
          train/
            patient_001/
              T2FS.npy       # [H, W]
              DWI.npy
              seg.npy        # integer mask [H, W], 0=background, 1=tumor
              quant.npy      # [3]: (x_center, y_center, area) normalized to [0,1]
            patient_002/
              ...
          val/   (same structure)
          test/  (same structure, quant.npy optional)

    When label_ratio < 1.0, only that fraction of training samples has
    seg / quant labels; the rest are returned with masks of -1 (ignored).
    """

    # Non-contrast modalities fed to the student
    STUDENT_MODALITIES = ['T2FS', 'DWI']
    # CE-MRI modalities fed to the teacher (any one present is used)
    TEACHER_MODALITIES = ['CE', 'T1CE', 'arterial', 'venous']

    def __init__(self, root, mode='train', transforms=None, label_ratio=1.0,
                 load_ce=True):
        self.root = Path(root)
        self.mode = mode
        self.transforms = transforms
        self.label_ratio = label_ratio if mode == 'train' else 1.0
        self.load_ce = load_ce

        split_dir = self.root / mode
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        if not self.samples:
            raise RuntimeError(f"No patient directories found in {split_dir}")

        # Determine labeled subset
        n_labeled = max(1, int(len(self.samples) * self.label_ratio))
        self.labeled_ids = set(range(n_labeled))

    def __len__(self):
        return len(self.samples)

    def _load_normalize(self, path):
        arr = np.load(path).astype(np.float32)
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    def __getitem__(self, idx):
        patient_dir = self.samples[idx]
        is_labeled = idx in self.labeled_ids

        # Student input: non-contrast modalities (T2FS + DWI)
        image = np.stack(
            [self._load_normalize(patient_dir / f'{m}.npy') for m in self.STUDENT_MODALITIES],
            axis=0,
        )  # [2, H, W]

        # Teacher input: first available CE-MRI modality, or None
        ce_image = None
        if self.load_ce:
            for mod in self.TEACHER_MODALITIES:
                ce_path = patient_dir / f'{mod}.npy'
                if ce_path.exists():
                    ce_arr = self._load_normalize(ce_path)
                    # Replicate to match student channel count so teacher uses same arch
                    ce_image = np.stack([ce_arr] * len(self.STUDENT_MODALITIES), axis=0)
                    break

        seg = (np.load(patient_dir / 'seg.npy').astype(np.int64)
               if is_labeled
               else np.full(image.shape[1:], -1, dtype=np.int64))

        quant_path = patient_dir / 'quant.npy'
        quant = (np.load(quant_path).astype(np.float32)
                 if (is_labeled and quant_path.exists())
                 else np.full(3, -1.0, dtype=np.float32))

        sample = {
            'image': image,
            'seg': seg,
            'quant': quant,
            'id': patient_dir.name,
            'labeled': is_labeled,
            'has_ce': ce_image is not None,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)
            if ce_image is not None:
                # Apply same spatial transforms to CE image (reuse ToTensor only)
                ce_sample = {'image': ce_image, 'seg': seg, 'quant': quant}
                ce_sample = self.transforms(ce_sample)
                sample['ce_image'] = ce_sample['image']
            else:
                sample['ce_image'] = torch.zeros_like(sample['image'])
        else:
            sample['image'] = torch.from_numpy(image)
            sample['seg'] = torch.from_numpy(seg)
            sample['quant'] = torch.from_numpy(quant)
            sample['ce_image'] = (torch.from_numpy(ce_image)
                                  if ce_image is not None
                                  else torch.zeros_like(sample['image']))

        return sample
