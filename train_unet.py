import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from unet_model import build_unet

# === Paths ===
image_dir = "../patches/images"
mask_dir = "../patches/masks"
results_dir = "./results/predictions"
model_save_path = "./results/model_weights.h5"

# === Load all images and masks ===
def load_data():
    image_files = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))

    X = []
    y = []

    for img_path, msk_path in zip(image_files, mask_files):
        img = np.array(Image.open(img_path)) / 255.0  # Normalize RGB
        msk = np.array(Image.open(msk_path)) / 255.0  # Binary mask (0 or 1)

        X.append(img)
        y.append(msk)

    X = np.array(X).astype(np.float32)
    y = np.expand_dims(np.array(y).astype(np.float32), axis=-1)  # Shape: (H, W, 1)
    return X, y

# === Load and split data ===
X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# === Build model ===
model = build_unet(input_shape=(128, 128, 3))
model.compile(optimizer=Adam(1e-4),
              loss=BinaryCrossentropy(),
              metrics=[BinaryAccuracy()])

# === Train ===
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,
                    batch_size=16)

# === Save model weights ===
model.save(model_save_path)
print(f"✅ Model saved to {model_save_path}")

# === Save predictions on validation set ===
os.makedirs(results_dir, exist_ok=True)

for i in range(5):
    pred = model.predict(np.expand_dims(X_val[i], axis=0))[0, :, :, 0]
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    Image.fromarray(pred_mask).save(os.path.join(results_dir, f"prediction_{i+1}.png"))
    Image.fromarray((y_val[i, :, :, 0] * 255).astype(np.uint8)).save(os.path.join(results_dir, f"groundtruth_{i+1}.png"))

print("✅ Sample predictions saved to:", results_dir)