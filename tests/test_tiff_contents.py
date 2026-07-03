import os
from PIL import Image
import numpy as np

tiff_path = r"C:\Users\m\Projects\rtiDb\server\uploads\neural_mona.tif"

if not os.path.exists(tiff_path):
    print("TIFF not found on host!")
    exit(1)

img = Image.open(tiff_path)
print("Format:", img.format)
print("Mode:", img.mode)
print("Size:", img.size)

data = np.array(img)
print("Data shape:", data.shape)

H, W, C = data.shape
flat_data = data.reshape(-1, C)

small_alpha_indices = np.where(flat_data[:, 3] < 10)[0]
print(f"Number of pixels with Alpha < 10: {len(small_alpha_indices)}")

if len(small_alpha_indices) > 0:
    print("Sample pixels (R, G, B, A) with small Alpha:")
    for idx in small_alpha_indices[:10]:
        print(flat_data[idx])
        
non_zero_rgb_with_small_alpha = np.any((flat_data[:, :3] > 0) & (flat_data[:, 3:4] < 5))
print("Is there any pixel with R/G/B > 0 when A < 5?", non_zero_rgb_with_small_alpha)
if non_zero_rgb_with_small_alpha:
    indices = np.where((np.any(flat_data[:, :3] > 0, axis=1)) & (flat_data[:, 3] < 5))[0]
    print(f"Found {len(indices)} such pixels. Samples:")
    for idx in indices[:5]:
        print(flat_data[idx])
