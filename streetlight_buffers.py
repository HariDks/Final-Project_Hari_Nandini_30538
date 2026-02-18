import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# --- Step 1: Load streetlight data ---
# Replace with your CSV path
df = pd.read_csv("chicago_streetlights_all.csv")  

# Make sure column names are correct
# Columns should be 'latitude' and 'longitude'
print(df.head())

# --- Step 2: Convert to GeoDataFrame ---
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84

# --- Step 3: Project to metric CRS (Chicago, meters) ---
# EPSG:3435 is suitable for Chicago
gdf = gdf.to_crs(epsg=3435)

# --- Step 4: Create buffers ---
buffer_sizes = [15, 20, 30]  # in meters
buffer_gdfs = []

for size in buffer_sizes:
    temp = gdf.copy()
    temp['buffer_radius_m'] = size
    temp['geometry'] = temp.geometry.buffer(size)
    buffer_gdfs.append(temp)

# Combine all buffers into one GeoDataFrame
all_buffers_gdf = gpd.GeoDataFrame(pd.concat(buffer_gdfs, ignore_index=True), crs=gdf.crs)

# --- Step 5: Save to GeoJSON ---
all_buffers_gdf.to_file("streetlights_buffers.geojson", driver="GeoJSON")



