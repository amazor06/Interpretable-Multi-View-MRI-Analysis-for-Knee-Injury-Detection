import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load ACL labels
# -----------------------------
labels = pd.read_csv("CSV Files/train-acl.csv", header=None)
labels.columns = ["exam_id", "acl_label"]

# Pick one untorn and one torn exam
untorn_id = int(labels[labels["acl_label"] == 0].iloc[0]["exam_id"])
torn_id = int(labels[labels["acl_label"] == 1].iloc[0]["exam_id"])

# -----------------------------
# Load MRI volumes
# -----------------------------
untorn = np.load(f"data/train/sagittal/{untorn_id:04d}.npy")
torn = np.load(f"data/train/sagittal/{torn_id:04d}.npy")

print("Untorn exam:", untorn_id, untorn.shape)
print("Torn exam:", torn_id, torn.shape)

num_slices = min(untorn.shape[0], torn.shape[0])

# -----------------------------
# Pause control
# -----------------------------
paused = False

def toggle_pause(event):
    global paused
    paused = not paused

# -----------------------------
# Create figure
# -----------------------------
fig = plt.figure(figsize=(12, 6))

# Mouse click pauses/unpauses
fig.canvas.mpl_connect('button_press_event', toggle_pause)

# Keyboard spacebar pauses/unpauses
fig.canvas.mpl_connect(
    'key_press_event',
    lambda event: toggle_pause(event) if event.key == ' ' else None
)

# -----------------------------
# Scroll through slices
# -----------------------------
i = 0

while i < num_slices:

    if not paused:
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.imshow(untorn[i], cmap="gray")
        plt.title(f"Untorn ACL\nExam {untorn_id}, Slice {i}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(torn[i], cmap="gray")
        plt.title(f"Torn ACL\nExam {torn_id}, Slice {i}")
        plt.axis("off")

        plt.tight_layout()

        i += 1

    plt.pause(0.2)
plt.show()