import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pre_path  = "idabd/images/AOI1-tile_1-3_pre_disaster.png"
post_path = "idabd/images/AOI1-tile_1-3_post_disaster.png"
mask_path = "idabd/masks/AOI1-tile_1-3_post_disaster.png"  # <-- change this


pre  = np.array(Image.open(pre_path).convert("RGB"))
post = np.array(Image.open(post_path).convert("RGB"))
mask_img = Image.open(mask_path)

mask = np.array(mask_img)

print("mask shape:", mask.shape, "dtype:", mask.dtype, "unique values:", np.unique(mask)[:20])

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(pre)
plt.axis("off")
plt.title("(a) Pre-disaster")

plt.subplot(1,3,2)
plt.imshow(post)
plt.axis("off")
plt.title("(b) Post-disaster")

plt.subplot(1,3,3)
# If mask is single-channel with values 0..4, this will look like the common “purple background / yellow buildings”
if mask.ndim == 2:
    plt.imshow(mask)  # default colormap often resembles the paper-style look
else:
    # If your mask is RGB-coded already, just display it
    plt.imshow(mask)
plt.axis("off")
plt.title("(c) Building damage annotation")

plt.tight_layout()
plt.show()
