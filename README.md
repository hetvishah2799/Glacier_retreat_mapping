# Glacier_retreat_mapping
Deep Learning- based glacier retreat mapping using Sentinel-2 imagery
# Glacier_retreat_mapping 🌍🧊

**Deep Learning‑based glacier retreat mapping using Sentinel‑2 imagery**

---

## 📘 Overview

This project presents a UNet-based deep learning workflow to map how Gangotri Glacier (Himalayas) has retreated between 2016 and 2024 using Sentinel‑2 data. It includes:

- Patch-based training  
- Full image inference  
- Time-series retreat graph  
- Animated timelapse  
- Final summary after‑map

---

## 🗂️ Folder Structure

GlacierMappingProjectClean/
├── Data/
│ ├── Glacier_RGB_2016.tif* ← Sample RGB (before glacier retreat)
│ ├── Glacier_RGB_2024.tif* ← Sample RGB (after glacier retreat)
│ ├── glacier_2016_prediction.tif* ← Mask predicted by model (2016)
│ ├── glacier_2024_prediction.tif* ← Mask predicted by model (2024)
│ └── glacier_aoi_merged.shp ← Study area polygon
│
├── Model/
│ ├── train_unet.py ← Model training script (U‑Net architecture)
│ └── unet_model.py ← U‑Net definition module
│
├── Full_image_inference/
│ ├── predict_full_image.py ← Inference for one image
│ └── predict_years_loop.py ← Inference loop for multiple years
│
├── results/
│ ├── glacier_area_by_year.csv ← Glacier area (km²) per year
│ ├── glacier_retreat_graph.png ← Area vs year plot
│ ├── glacier_retreat_timelapse.gif ← Animated retreat timelapse
│
├── Patches/ ← Sample training patches (images + masks)
├── generate_patches.py ← Creates training data patches
├── plot_glacier_retreat.py ← Generates the retreat plot
├── make_retreat_timelapse.py ← Builds visual timelapse animation
├── Calculate_glacier_area.py ← Converts mask to area stats
├── README.md ← (this file)
└── .gitignore ← Ignores large files (.tif, .h5)


\* These are **sample files only**. Large files (>100MB) are omitted to make the repo lightweight.  
Use the instructions below to add full-resolution data.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/hetvishah2799/Glacier_retreat_mapping.git
cd Glacier_retreat_mapping

2. (Optional) Add large data files
Download and place in Data/:

Glacier_RGB_2016.tif (~120MB)

Glacier_RGB_2024.tif (~120MB)

Full-resolution glacier predictions

(You can link these externally via Google Drive—check LICENSE section.)

3. Install dependencies

pip install -r requirements.txt

(Create a requirements.txt file with tensorflow, rasterio, geopandas, matplotlib, pillow)

4. Train the U‑Net model (optional)

python train_unet.py --epochs 10

5. Run inference

python predict_full_image.py --input Data/Glacier_RGB_2016.tif --output predictions/glacier_2016_prediction.tif
python predict_full_image.py --input Data/Glacier_RGB_2024.tif --output predictions/glacier_2024_prediction.tif

6. Produce visual results
Compute yearly glacier areas:
python Calculate_glacier_area.py

Generate area vs year plot:
python plot_glacier_retreat.py

Build S2 timelapse:
python make_retreat_timelapse.py

Generated files are in the results/ folder.
