import numpy as np
import rasterio
from rasterio.windows import Window
from tensorflow.keras.models import load_model
import os

# === Paths ===
rgb_path = "./data/processed/Glacier_RGB_2024.tif"
model_path = "./model/results/model_weights.h5"
output_dir = "./Full_image_inference"
output_path = os.path.join(output_dir, "Glacier_2024_prediction.tif")

PATCH_SIZE = 128

# === Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# === Load model ===
print("⏳ Loading model from:", model_path)
model = load_model(model_path)
print("✅ Model loaded successfully!")

# === Read input image ===
with rasterio.open(rgb_path) as src:
    profile = src.profile
    width, height = src.width, src.height
    transform = src.transform

    print(f"🖼  Image size: width={width}, height={height}")

    # Prepare empty output array
    prediction_full = np.zeros((height, width), dtype=np.uint8)
    patch_count = 0

    # === Sliding window over full image
    for row in range(0, height, PATCH_SIZE):
        for col in range(0, width, PATCH_SIZE):
            if row + PATCH_SIZE > height or col + PATCH_SIZE > width:
                continue

            print(f"📦 Processing patch at row={row}, col={col}")

            patch = src.read(window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
            patch = np.moveaxis(patch, 0, -1)
            patch = (patch / 255.0).astype(np.float32)

            if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3):
                print(f"⚠ Skipping patch due to shape mismatch: {patch.shape}")
                continue

            pred = model.predict(np.expand_dims(patch, axis=0), verbose=0)[0, :, :, 0]
            pred_mask = (pred > 0.5).astype(np.uint8)

            prediction_full[row:row+PATCH_SIZE, col:col+PATCH_SIZE] = pred_mask * 255
            patch_count += 1

print(f"✅ Prediction complete for {patch_count} patches.")
print("💾 Saving glacier mask to:", os.path.abspath(output_path))

# === Save final glacier mask
profile.update({
    'count': 1,
    'dtype': 'uint8',
    'compress': 'lzw'
})

with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(prediction_full, 1)

print("✅ Glacier mask saved successfully!")