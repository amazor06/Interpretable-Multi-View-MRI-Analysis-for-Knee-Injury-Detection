import os
import numpy as np

def normalize_dataset(input_root, output_root):

    planes = ["sagittal", "coronal", "axial"]

    for plane in planes:

        input_dir = f"{input_root}/{plane}"
        output_dir = f"{output_root}/{plane}"

        os.makedirs(output_dir, exist_ok=True)

        files = sorted(os.listdir(input_dir))

        print(f"\nProcessing {plane}")
        print(f"Files: {len(files)}")

        for i, filename in enumerate(files):

            if filename.endswith(".npy"):

                volume = np.load(
                    os.path.join(input_dir, filename)
                )

                volume = volume.astype(np.float32)
                volume = volume / 255.0

                np.save(
                    os.path.join(output_dir, filename),
                    volume
                )

                if i % 100 == 0:
                    print(f"{i}/{len(files)}")

# -----------------------------
# Normalize training set
# -----------------------------
normalize_dataset(
    "data/train",
    "data_normalized/train"
)

# -----------------------------
# Normalize validation set
# -----------------------------
normalize_dataset(
    "data/valid",
    "data_normalized/valid"
)

print("\nDone.")