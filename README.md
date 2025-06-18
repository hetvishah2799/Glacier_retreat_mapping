# 🧊 Glacier Retreat Mapping using Deep Learning & Sentinel-2 Imagery

This project demonstrates how to detect and visualize *glacier retreat* over time using *Sentinel‑2 satellite images, deep learning (U‑Net), and open-source geospatial tools like **QGIS* and *Python*.

📍 *Study Area:* Gangotri Glacier, Uttarakhand, India  
📆 *Time Range:* 2016–2024  
🛰 *Satellite Data:* Sentinel‑2 (L2A, atmospherically corrected)

---

## 📌 Objectives

- Create glacier masks using deep learning segmentation (U‑Net)
- Analyze glacier area change over multiple years
- Visualize retreat using graphs and animated maps
- Publish the full workflow for reproducibility

---

## 🧠 Methodology

1. *Data Collection*  
   - Downloaded Sentinel‑2 imagery (bands B2, B3, B4) from Copernicus Hub  
   - Selected cloud-free images from *2016 to 2024* (1 image/year)

2. *Preprocessing*
   - Merged and clipped Sentinel‑2 tiles to the AOI (Gangotri Glacier) in QGIS  
   - Created glacier outlines using RGI glacier shapefiles  
   - Generated glacier masks to train the model

3. *Patch Generation*
   - Divided RGB images and masks into small 128×128 px patches  
   - Used patches to train a semantic segmentation model

4. *Model Training*
   - Used a *U‑Net architecture* in TensorFlow/Keras  
   - Trained the model on glacier/non-glacier classification patches

5. *Full Image Prediction*
   - Applied the trained U‑Net model to full satellite images  
   - Predicted glacier extent for each year (2016–2024)

6. *Postprocessing & Analysis*
   - Converted predicted binary masks to glacier area (in km²)  
   - Plotted glacier retreat trend over time  
   - Generated a *glacier retreat timelapse GIF*

---

## 🔧 Tools & Libraries

- 🛰 *Sentinel-2 data* (Copernicus Open Access Hub)
- 🗺 *QGIS* – AOI extraction, raster clipping, shapefile creation
- 🐍 *Python Libraries*  
  - rasterio, numpy, matplotlib, tensorflow, geopandas, PIL

---

## 🧪 How to Reproduce This Project

### Step 1: Install dependencies
```bash
pip install numpy rasterio matplotlib geopandas pillow tensorflow

### Step 2: Download Sentinel‑2 imagery (L2A) from Copernicus Hub
```bash
Bands required: B2 (blue), B3 (green), B4 (red)

Select 1 cloud-free image per year for your glacier AOI


### Step 3: Preprocess images in QGIS
```bash
Clip tiles to your glacier boundary shapefile (AOI)

Merge if AOI spans multiple tiles

Export as single RGB .tif per year


### Step 4: Generate training patches
```bash
python generate_patches.py

### Step 5: Train the U‑Net model
```bash
python train_unet.py

### Step 6: Predict glacier masks
```bash
python predict_full_image.py

### Step 7: Analyze area
```bash
python Calculate_glacier_area.py

### Step 8: Plot & visualize
```bash
python plot_glacier_retreat.py
python make_retreat_timelapse.py


---

### 📊 Example Outputs (what this project generates)

✅ glacier_area_by_year.csv — glacier area per year

✅ glacier_retreat_graph.png — glacier loss over time

✅ glacier_retreat_timelapse.gif — visual animation of glacier change



---

### 🌐 Data Sources

Sentinel‑2 imagery: Copernicus Open Access Hub

Glacier outlines: RGI (Randolph Glacier Inventory)

AOI shapefile: Digitized from QGIS over Gangotri Glacier


---

### 🙋 Author

Hetvi Shah
MSc in Geoinformatics | Deep Learning for Remote Sensing
🔗 LinkedIn Profile
📧 hetvishah2799@gmail.com


---

### 📜 License

This project is licensed under the MIT License.
You may reuse, modify, and extend it with credit.
