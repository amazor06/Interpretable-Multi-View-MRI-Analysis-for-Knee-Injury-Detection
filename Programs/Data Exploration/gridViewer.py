import numpy as np
import matplotlib.pyplot as plt

volume = np.load("data/train/sagittal/0000.npy")

fig, axes = plt.subplots(5, 6, figsize=(12, 10))

for i, ax in enumerate(axes.flat): 
    ax.imshow(volume[i], cmap='gray')
    ax.set_title(f"{i}")
    ax.axis('off')

plt.tight_layout
plt.show()
