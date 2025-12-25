import os
import nibabel as nib
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = nib.load(path).get_fdata().astype("float32")

        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-5)

        # Shape: 1 × D × H × W
        img = torch.tensor(img).unsqueeze(0)

        # TEMPORARY LABEL
        # Replace this with real labels later
        label = 0

        return img, label
