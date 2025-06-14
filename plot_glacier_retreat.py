import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv("glacier_area_by_year.csv")

# Use seaborn style for modern look
plt.style.use("seaborn-v0_8-whitegrid")

# Plot setup
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["Year"], df["Glacier_Area_km2"], marker='o', markersize=8,
        linewidth=2.5, color="#1f77b4", label="Glacier Area")

# Add area values as labels on points
for x, y in zip(df["Year"], df["Glacier_Area_km2"]):
    ax.text(x, y + 0.05, f"{y:.1f}", ha='center', fontsize=10, color="#333333")

# Title and labels
ax.set_title("🧊 Glacier Retreat Trend at Gangotri (2016–2023)", fontsize=16, fontweight='bold')
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Glacier Area (km²)", fontsize=13)

# Y-axis range (customizable to flatten dips)
ax.set_ylim(1490, 1510)  # <- Try different ranges to make dip look smoother

# Customize ticks and grid
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, which='major', linestyle='--', alpha=0.5)

# Optional: add background color
ax.set_facecolor("#f8f9fa")

# Save high-res plot
plt.tight_layout()
plt.savefig("results/glacier_retreat_styled.png", dpi=300)
plt.show()