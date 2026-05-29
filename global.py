import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_DIR = "CSV Files"
DATA_DIR = "data/train"

# -----------------------------
# Load labels
# -----------------------------
acl_df = pd.read_csv(f"{CSV_DIR}/train-acl.csv", header=None)
acl_df.columns = ["exam_id", "label"]

meniscus_df = pd.read_csv(f"{CSV_DIR}/train-meniscus.csv", header=None)
meniscus_df.columns = ["exam_id", "label"]

abnormal_df = pd.read_csv(f"{CSV_DIR}/train-abnormal.csv", header=None)
abnormal_df.columns = ["exam_id", "label"]

# -----------------------------
# Label statistics
# -----------------------------
print("\n=== LABEL COUNTS ===")

datasets = {
    "ACL": acl_df,
    "Meniscus": meniscus_df,
    "Abnormal": abnormal_df
}

for name, df in datasets.items():

    healthy = (df["label"] == 0).sum()
    injured = (df["label"] == 1).sum()

    print(f"\n{name}")
    print(f"Healthy: {healthy}")
    print(f"Positive/Injured: {injured}")

    total = len(df)

    print(f"Healthy %: {100 * healthy / total:.2f}")
    print(f"Positive %: {100 * injured / total:.2f}")

# -----------------------------
# Analyze MRI shapes
# -----------------------------
print("\n=== MRI SHAPE ANALYSIS ===")

planes = ["sagittal", "coronal", "axial"]

shape_data = {
    "sagittal": [],
    "coronal": [],
    "axial": []
}

for exam_id in acl_df["exam_id"]:

    for plane in planes:

        path = f"{DATA_DIR}/{plane}/{exam_id:04d}.npy"

        volume = np.load(path)

        num_slices = volume.shape[0]

        shape_data[plane].append(num_slices)

# -----------------------------
# Print statistics
# -----------------------------
for plane in planes:

    slices = np.array(shape_data[plane])

    print(f"\n{plane.upper()}")

    print(f"Min slices: {slices.min()}")
    print(f"Max slices: {slices.max()}")
    print(f"Mean slices: {slices.mean():.2f}")
    print(f"Std dev: {slices.std():.2f}")

# -----------------------------
# Histograms
# -----------------------------
plt.figure(figsize=(12, 5))

for i, plane in enumerate(planes):

    plt.subplot(1, 3, i + 1)

    plt.hist(shape_data[plane], bins=15)

    plt.title(f"{plane} slice counts")
    plt.xlabel("Number of slices")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()