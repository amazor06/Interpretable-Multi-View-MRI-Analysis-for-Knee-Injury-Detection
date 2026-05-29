import numpy as np
import matplotlib.pyplot as plt 

volume = np.load("data/train/sagittal/0200.npy")

plt.figure(figsize= (10,10))

for i in range(volume.shape[0]): 
    plt.clf()

    plt.imshow(volume[i], cmap='gray')
    plt.title(f"Slice {i}")

    plt.axis('off')
    plt.pause(0.2)

plt.show()