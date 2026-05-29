import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# -----------------------------
# Settings
# -----------------------------
TRAIN_CSV = "CSV Files/train-acl.csv"
VALID_CSV = "CSV Files/valid-acl.csv"

TRAIN_DIR = "data_normalized/train/sagittal"
VALID_DIR = "data_normalized/valid/sagittal"

MODEL_OUT = "models/acl_sagittal_cnn.pth"

NUM_SLICES = 16
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4

os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Helper: extract center slices
# -----------------------------
def center_slice_window(volume, num_slices=16):
    """
    Converts variable-depth MRI volume into fixed number of center slices.
    Input shape:  (D, H, W)
    Output shape: (num_slices, H, W)
    """

    depth = volume.shape[0]
    center = depth // 2
    half = num_slices // 2

    start = center - half
    end = start + num_slices

    # Pad if scan has too few slices
    if start < 0 or end > depth:
        pad_before = max(0, -start)
        pad_after = max(0, end - depth)

        volume = np.pad(
            volume,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant"
        )

        start = start + pad_before
        end = end + pad_before

    return volume[start:end]


# -----------------------------
# Dataset class
# -----------------------------
class MRNetACLDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.labels = pd.read_csv(csv_path, header=None)
        self.labels.columns = ["exam_id", "label"]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        exam_id = int(self.labels.iloc[idx]["exam_id"])
        label = float(self.labels.iloc[idx]["label"])

        file_path = os.path.join(self.data_dir, f"{exam_id:04d}.npy")

        volume = np.load(file_path).astype(np.float32)

        # If not already normalized, normalize here safely
        if volume.max() > 1.0:
            volume = volume / 255.0

        volume = center_slice_window(volume, NUM_SLICES)

        # Shape becomes: [channels, height, width]
        # We treat 16 MRI slices as 16 input channels.
        x = torch.tensor(volume, dtype=torch.float32)

        y = torch.tensor(label, dtype=torch.float32)

        return x, y


# -----------------------------
# CNN Model
# -----------------------------
class ACLCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(NUM_SLICES, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


# -----------------------------
# Load data
# -----------------------------
train_dataset = MRNetACLDataset(TRAIN_CSV, TRAIN_DIR)
valid_dataset = MRNetACLDataset(VALID_CSV, VALID_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_dataset))
print("Valid samples:", len(valid_dataset))


# -----------------------------
# Handle class imbalance
# -----------------------------
train_labels = train_dataset.labels["label"].values
num_negative = np.sum(train_labels == 0)
num_positive = np.sum(train_labels == 1)

pos_weight_value = num_negative / num_positive
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

print("Negative:", num_negative)
print("Positive:", num_positive)
print("Positive class weight:", pos_weight_value)


# -----------------------------
# Model, loss, optimizer
# -----------------------------
model = ACLCNN().to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -----------------------------
# Training + validation loop
# -----------------------------
best_auc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Training
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, preds)

    print("Train loss:", round(train_loss, 4))
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall/Sensitivity:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion matrix:")
    print(cm)

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), MODEL_OUT)
        print("Saved best model.")

print("\nTraining complete.")
print("Best validation ROC-AUC:", round(best_auc, 4))
print("Saved model:", MODEL_OUT)