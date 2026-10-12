import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.mri_dataset import MRIDataset
from models.mri_model import MRI3DNet


def parse_args():
    p = argparse.ArgumentParser(description="Train a 3D CNN on NIfTI MRI volumes.")
    p.add_argument("--data", default="data", help="folder containing train/ and val/")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--out", default="checkpoints/best.pt")
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for imgs, labels in tqdm(loader, desc="train" if training else "val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += imgs.size(0)

    return total_loss / seen, correct / seen


def main():
    args = parse_args()
    device = get_device()

    train_ds = MRIDataset(os.path.join(args.data, "train"))
    val_ds = MRIDataset(os.path.join(args.data, "val"))
    print(f"Device: {device} | train: {len(train_ds)} | val: {len(val_ds)}")
    for name, ds in (("train", train_ds), ("val", val_ds)):
        if ds.unlabeled:
            print(f"  skipping {len(ds.unlabeled)} unlabeled {name} volumes (add them to labels.csv)")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(
            "No labeled volumes found. Put .nii/.nii.gz files in data/train and data/val, "
            "each with a labels.csv (columns: filename,label)."
        )

    # batch_size=1: full-resolution 3D volumes are memory-heavy
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = MRI3DNet(num_classes=args.num_classes).to(device)
    # Dry run to materialize the LazyLinear head before the optimizer collects parameters
    with torch.no_grad():
        model(train_ds[0][0].unsqueeze(0).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} acc {val_acc:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out)
            print(f"  saved best model → {args.out}")


if __name__ == "__main__":
    main()
