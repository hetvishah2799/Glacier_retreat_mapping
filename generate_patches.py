import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

# === Paths ===
image_path = "data/processed/Glacier_RGB_clipped.tif"
mask_path = "data/processed/glacier_mask.tif"
output_img_dir = "patches/images"
output_mask_dir = "patches/masks"

# === Patch size ===
PATCH_SIZE = 128
STRIDE = 128  # set to smaller (like 64) if you want overlapping patches

# === Create output folders if not exist ===
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# === Open both image and mask ===
with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
    assert img_src.width == mask_src.width and img_src.height == mask_src.height, "Size mismatch"
    
    width, height = img_src.width, img_src.height
    patch_id = 0

    for row in range(0, height, STRIDE):
        for col in range(0, width, STRIDE):
            if row + PATCH_SIZE > height or col + PATCH_SIZE > width:
                continue  # skip incomplete patches at edges
            
            # Read RGB patch
            img_patch = img_src.read(window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
            img_patch = np.moveaxis(img_patch, 0, -1)  # Convert from (bands, H, W) to (H, W, bands)

            # Read mask patch
            mask_patch = mask_src.read(1, window=Window(col, row, PATCH_SIZE, PATCH_SIZE))

            # OPTIONAL: Skip patches with no glacier at all
            if np.sum(mask_patch) == 0:
                continue

            # Save image patch
            # Normalize to 0–255
            img_norm = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min())
            img_scaled = (img_norm * 255).astype(np.uint8)
            img = Image.fromarray(img_scaled)
            img.save(os.path.join(output_img_dir, f"patch_{patch_id:04d}.png"))

            # Save mask patch
            msk = Image.fromarray((mask_patch * 255).astype(np.uint8))  # convert 0/1 to 0/255
            msk.save(os.path.join(output_mask_dir, f"patch_{patch_id:04d}.png"))

            patch_id += 1

print(f"✅ {patch_id} patches saved to 'patches/images' and 'patches/masks'")