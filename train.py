import torch
from torch.utils.data import DataLoader
from dataset.mri_dataset import MRIDataset
from models.mri_model import MRI3DNet
from tqdm import tqdm

# Device setup
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load datasets
train_dataset = MRIDataset("data/train")
val_dataset = MRIDataset("data/val")

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Device:", device)

# Safety check
if len(train_dataset) == 0 or len(val_dataset) == 0:
    print("\nERROR: Dataset folders are empty.")
    print("Add .nii or .nii.gz MRI files to data/train and data/val.")
    exit()

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# Initialize model, optimizer, loss
model = MRI3DNet(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Placeholder labels since dataset is not labeled yet
# Every sample gets label 0 for now


def generate_labels(batch_size):
    return torch.zeros(batch_size, dtype=torch.long)


# Training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(device)

        labels = generate_labels(imgs.size(0)).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} — Train Loss: {avg_loss:.4f}")

    # Validation (only checks forward pass)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            labels = generate_labels(imgs.size(0)).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} — Val Loss: {val_loss:.4f}")

print("\nTraining complete.")
