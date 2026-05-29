import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

labels = pd.read_csv("CSV Files/train-acl.csv", header=None)
labels.columns = ["exam_id", "acl_label"]

torn_id = labels[labels["acl_label"] == 1].iloc[0]["exam_id"]
untorn_id = labels[labels["acl_label"] == 0].iloc[0]["exam_id"]

print("Torn ACL exam:", torn_id)
print("Untorn ACL exam:", untorn_id)

torn = np.load(f"data/train/sagittal/{torn_id:04d}.npy")
untorn = np.load(f"data/train/sagittal/{untorn_id:04d}.npy")

print("Torn shape:", torn.shape)
print("Untorn shape:", untorn.shape)

torn_slice = torn.shape[0] // 2
untorn_slice = untorn.shape[0] // 2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(untorn[untorn_slice], cmap="gray")
axes[0].set_title(f"Untorn ACL\nExam {untorn_id}, Slice {untorn_slice}")
axes[0].axis("off")

axes[1].imshow(torn[torn_slice], cmap="gray")
axes[1].set_title(f"Torn ACL\nExam {torn_id}, Slice {torn_slice}")
axes[1].axis("off")

plt.tight_layout()
plt.show()