import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
from rasterio.plot import plotting_extent
from rasterio.warp import reproject, Resampling

# === CONFIG ===
years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
rgb_image_path = "data/processed/Glacier_RGB_2020.tif"
aoi_path = "data/processed/glacier_aoi_merged.shp"
mask_folder = "full_image_inference"
gif_output = "results/glacier_retreat_timelapse.gif"

# === LOAD RGB IMAGE ===
with rasterio.open(rgb_image_path) as src:
    rgb_raw = np.moveaxis(src.read([1, 2, 3]), 0, -1).astype(np.float32)
    rgb_scaled = (rgb_raw -
    rgb_raw.min()) / (rgb_raw.max()- rgb_raw.min()) * 255
    rgb = np.clip(rgb_scaled, 0, 255).astype(np.uint8)
    rgb_extent = plotting_extent(src)
    crs = src.crs
    rgb_transform = src.transform
    height, width = src.height, src.width

# === LOAD AOI ===
aoi = gpd.read_file(aoi_path).to_crs(crs)

frames = []

for year in years:
    mask_path = os.path.join(mask_folder, f"glacier_{year}_prediction.tif")
    if not os.path.exists(mask_path):
        print(f"❌ Skipping {year} — mask not found.")
        continue

    # === Reproject and resample mask to match RGB ===
    with rasterio.open(mask_path) as mask_src:
        mask_data = mask_src.read(1)
        mask_meta = mask_src.meta.copy()

        # Prepare output array (same size as RGB)
        mask_resampled = np.zeros((height, width), dtype=np.uint8)

        reproject(
            source=mask_data,
            destination=mask_resampled,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=rgb_transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )

    # === Create blue transparent overlay ===
    mask = (mask_resampled > 127).astype(np.uint8)
    blue_overlay = np.zeros((height, width, 4), dtype=np.uint8)
    blue_overlay[:, :, 2] = 255  # Blue channel
    blue_overlay[:, :, 3] = mask * 120  # Alpha transparency

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(rgb, extent=rgb_extent)
    ax.imshow(blue_overlay, extent=rgb_extent)

    # AOI
    aoi.boundary.plot(ax=ax, edgecolor='yellow', linewidth=1.5)

    # Year label
    ax.text(0.05, 0.92, f"Year: {year}", transform=ax.transAxes,
            fontsize=18, color='white', weight='bold',
            bbox=dict(facecolor='black', alpha=0.6))

    ax.axis('off')
    plt.tight_layout()

    # Save frame
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = Image.fromarray(img[:, :, :3])  # RGB only
    frames.append(img)
    plt.close()

# === Save as GIF ===
os.makedirs("results", exist_ok=True)
frames[0].save(gif_output, save_all=True, append_images=frames[1:], duration=1200, loop=0)
print(f"✅ Glacier retreat timelapse saved: {gif_output}")