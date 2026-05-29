import os
import pandas as pd
import numpy as np

CSV_DIR = "CSV Files"
DATA_DIR = "data/train"

label_files = {
    "ACL": "train-acl.csv",
    "Meniscus": "train-meniscus.csv",
    "Abnormal": "train-abnormal.csv"
}

print("=== Label Summary ===")

for label_name, csv_file in label_files.items():
    path = os.path.join(CSV_DIR, csv_file)
    df = pd.read_csv(path, header=None)
    df.columns = ["exam_id", "label"]

    print(f"\n{label_name}")
    print("Total exams:", len(df))
    print(df["label"].value_counts().sort_index())

print("\n=== MRI Volume Shape Check ===")

acl_df = pd.read_csv(os.path.join(CSV_DIR, "train-acl.csv"), header=None)
acl_df.columns = ["exam_id", "acl_label"]

sample_ids = acl_df["exam_id"].head(5)

for exam_id in sample_ids:
    print(f"\nExam {exam_id:04d}")

    for plane in ["sagittal", "coronal", "axial"]:
        file_path = os.path.join(DATA_DIR, plane, f"{exam_id:04d}.npy")

        if os.path.exists(file_path):
            volume = np.load(file_path)
            print(f"{plane}: shape={volume.shape}, dtype={volume.dtype}, min={volume.min()}, max={volume.max()}")
        else:
            print(f"{plane}: missing file")