import numpy as np
import matplotlib.pyplot as plt

volume = np.load("data/train/sagittal/0004.npy")

slice_idx = 15

binary = volume[slice_idx] > 100

plt.imshow(binary, cmap='gray')
plt.title("Thresholded MRI")
plt.axis('off')

plt.show()