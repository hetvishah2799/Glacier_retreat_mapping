import os
import rasterio
import numpy as np

# Folder containing glacier prediction masks
folder = "full_image_inference"
output_csv = "glacier_area_by_year.csv"

# Years to process
years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,2024]

# Store results
results = []

for year in years:
    path = os.path.join(folder, f"glacier_{year}_prediction.tif")
    with rasterio.open(path) as src:
        data = src.read(1)
        pixel_size = src.res[0] * src.res[1]  # should be 100 m²
        glacier_pixels = np.sum(data == 255)
        area_km2 = glacier_pixels * pixel_size / 1e6
        results.append((year, round(area_km2, 3)))
        print(f"✅ {year}: {round(area_km2, 3)} km²")

# Save to CSV
with open(output_csv, "w") as f:
    f.write("Year,Glacier_Area_km2\n")
    for year, area in results:
        f.write(f"{year},{area}\n")

print(f"\n🎉 Done! Results saved to {output_csv}")