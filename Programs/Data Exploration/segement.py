import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

file_path = "data/train/sagittal/0500.npy"
volume = np.load(file_path)

slice_idx = 15  # change this until ACL is visible
img = volume[slice_idx]

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap="gray")
ax.set_title(f"Draw around possible ACL region - Slice {slice_idx}")
ax.axis("off")

mask = np.zeros(img.shape, dtype=np.uint8)

def onselect(verts):
    global mask

    path = Path(verts)
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]
    points = np.vstack((x.ravel(), y.ravel())).T

    mask_flat = path.contains_points(points)
    mask = mask_flat.reshape(img.shape).astype(np.uint8)

    ax.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.4)
    plt.draw()

    np.save(f"acl_mask_slice_{slice_idx}.npy", mask)
    print(f"Saved mask: acl_mask_slice_{slice_idx}.npy")

selector = PolygonSelector(ax, onselect)
plt.show()