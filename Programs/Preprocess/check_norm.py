import numpy as np
import matplotlib.pyplot as plt

# Load sample MRI
volume = np.load("data_normalized/valid/sagittal/1140.npy")

print("Shape:", volume.shape)
print("Dtype:", volume.dtype)

print("Min:", volume.min())
print("Max:", volume.max())
print("Mean:", volume.mean())
print("Std:", volume.std())

# Flatten intensities
pixels = volume.flatten()

# Plot histogram
plt.hist(pixels, bins=50)

plt.title("MRI Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.show()