# process_streetlights.py

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# -------------------------------
# Step 1: Define project paths
# -------------------------------
project_dir = Path(__file__).parent  # folder where this script is located
data_dir = project_dir / "data"      # data folder

input_csv = data_dir / "streetlights_chicago.csv"
output_geojson = data_dir / "streetlights_buffers.geojson"

# -------------------------------
# Step 2: Load CSV
# -------------------------------
df = pd.read_csv(input_csv)

# Check your columns
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    raise ValueError("CSV must have 'latitude' and 'longitude' columns.")

# -------------------------------
# Step 3: Convert to GeoDataFrame
# -------------------------------
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84

# -------------------------------
# Step 4: Project to metric CRS
# -------------------------------
# EPSG:3435 is suitable for Chicago (meters)
gdf = gdf.to_crs(epsg=3435)

# -------------------------------
# Step 5: Create buffers
# -------------------------------
buffer_sizes = [15, 30, 50]  # in meters
buffer_gdfs = []

for size in buffer_sizes:
    temp = gdf.copy()
    temp['buffer_radius_m'] = size
    temp['geometry'] = temp.geometry.buffer(size)
    buffer_gdfs.append(temp)

# Combine all buffers into a single GeoDataFrame
all_buffers_gdf = gpd.GeoDataFrame(pd.concat(buffer_gdfs, ignore_index=True), crs=gdf.crs)

# -------------------------------
# Step 6: Save GeoJSON
# -------------------------------
all_buffers_gdf.to_file(output_geojson, driver="GeoJSON")
print(f"Saved buffers for all streetlights to: {output_geojson}")
