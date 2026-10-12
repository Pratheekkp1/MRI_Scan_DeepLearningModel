import csv
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset


def load_labels(csv_path):
    """Read a `filename,label` CSV into a {filename: int} dict."""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="") as f:
        return {row["filename"]: int(row["label"]) for row in csv.DictReader(f)}


class MRIDataset(Dataset):
    """NIfTI volumes from a folder, z-score normalized, shaped 1 × D × H × W.

    Labels come from `labels.csv` (columns: filename,label) in the same folder.
    Volumes without a label are skipped.
    """

    def __init__(self, folder, labels_file="labels.csv"):
        self.folder = folder
        labels = load_labels(os.path.join(folder, labels_file))
        files = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        )
        self.samples = [(os.path.join(folder, f), labels[f]) for f in files if f in labels]
        self.unlabeled = [f for f in files if f not in labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = nib.load(path).get_fdata().astype("float32")

        img = (img - img.mean()) / (img.std() + 1e-5)
        img = torch.from_numpy(img).unsqueeze(0)

        return img, label
