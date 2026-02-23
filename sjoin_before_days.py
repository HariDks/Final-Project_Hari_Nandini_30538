# join_streetlights_crime_pre_day_buckets.py
# Spatiotemporal join:
# Crimes within streetlight buffers AND in non-overlapping day buckets
# 1–5 days BEFORE creation_date.
#
# Bucket k means crimes in:
# [creation_date - k days, creation_date - (k-1) days)
# (end-exclusive)

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# -----------------------
# Paths
# -----------------------
project_dir = Path(__file__).parent
data_dir = project_dir / "data"

buffers_file = data_dir / "streetlights_buffers.geojson"
crime_csv = data_dir/"crimes_2011_2018.csv"
output_file = data_dir / "streetlight_crime_pre_day_buckets_events.geojson"

# -----------------------
# Load streetlight buffers
# -----------------------
buffers = gpd.read_file(buffers_file)

# Rename date columns if needed
if "creation_date" not in buffers.columns and "creating_date" in buffers.columns:
    buffers = buffers.rename(columns={"creating_date": "creation_date"})
if "completion_date" not in buffers.columns and "completed_date" in buffers.columns:
    buffers = buffers.rename(columns={"completed_date": "completion_date"})

needed_buf = {"creation_date", "buffer_radius_m", "geometry", "service_request_number"}
missing_buf = needed_buf - set(buffers.columns)
if missing_buf:
    raise ValueError(f"Missing buffer columns: {missing_buf}")

buffers["creation_date"] = pd.to_datetime(buffers["creation_date"], errors="coerce")
buffers = buffers.dropna(subset=["creation_date"]).copy()

# CRS
if buffers.crs is None:
    raise ValueError("Buffers have no CRS. Recreate buffers with a CRS.")
if buffers.crs.to_epsg() != 3435:
    buffers = buffers.to_crs(epsg=3435)

# Stable request_id
buffers["request_id"] = buffers["service_request_number"].astype(str)

# -----------------------
# Load crimes
# -----------------------
crime_df = pd.read_csv(crime_csv)

needed_crime = {"id", "date", "latitude", "longitude"}
missing_crime = needed_crime - set(crime_df.columns)
if missing_crime:
    raise ValueError(f"Missing crime columns: {missing_crime}")

crime_df["crime_date"] = pd.to_datetime(crime_df["date"], errors="coerce")
crime_df = crime_df.dropna(subset=["crime_date", "latitude", "longitude"]).copy()

crime_geom = [Point(xy) for xy in zip(crime_df["longitude"], crime_df["latitude"])]
crime_gdf = gpd.GeoDataFrame(crime_df, geometry=crime_geom, crs="EPSG:4326").to_crs(epsg=3435)

# -----------------------
# Spatial join ONCE
# -----------------------
joined = gpd.sjoin(crime_gdf, buffers, how="inner", predicate="within")
print("After spatial join:", len(joined))

# Ensure datetime
joined["crime_date"] = pd.to_datetime(joined["crime_date"], errors="coerce")
joined["creation_date"] = pd.to_datetime(joined["creation_date"], errors="coerce")

# -----------------------
# Compute day bucket (1..5 days before)
# We bucket by floor-to-day difference:
# day_diff = (creation_date_floor - crime_date_floor).days
# Keep 1..5 only.
# -----------------------
creation_day = joined["creation_date"].dt.floor("D")
crime_day = joined["crime_date"].dt.floor("D")

joined["days_before_request"] = (creation_day - crime_day).dt.days

# Keep crimes strictly before the request day (>=1) and up to 5 days
joined = joined[(joined["days_before_request"] >= 1) & (joined["days_before_request"] <= 5)].copy()

# Optional: enforce "just that day" precisely using non-overlapping intervals
# This removes any ambiguity around times:
# bucket k = [creation_date - k days, creation_date - (k-1) days)
bucket_start = joined["creation_date"] - pd.to_timedelta(joined["days_before_request"], unit="D")
bucket_end = joined["creation_date"] - pd.to_timedelta(joined["days_before_request"] - 1, unit="D")

exact_mask = (joined["crime_date"] >= bucket_start) & (joined["crime_date"] < bucket_end)
joined = joined[exact_mask].copy()

joined["bucket_start"] = bucket_start[exact_mask]
joined["bucket_end"] = bucket_end[exact_mask]

print("After day-bucket filter (1..5):", len(joined))

# -----------------------
# Keep analysis-ready columns
# -----------------------
cols = [
    # crime
    "id",
    "primary_type",
    "crime_date",
    "year",
    "community_area",
    "beat",
    "district",
    "ward",

    # streetlight
    "request_id",
    "service_request_number",
    "buffer_radius_m",
    "creation_date",
    "status",

    # bucket info
    "days_before_request",  # 1..5 (this is your key analysis variable)
    "bucket_start",
    "bucket_end",

    # geometry (crime point)
    "geometry",
]
cols = [c for c in cols if c in joined.columns]
joined = joined[cols]

# -----------------------
# Save
# -----------------------
joined.to_file(output_file, driver="GeoJSON")
print(f"Saved bucketed pre-day dataset to: {output_file}")

# -----------------------
# Sanity checks
# -----------------------
print("\nPreview:")
print(joined.head(3))

print("\nCounts by days_before_request (1..5):")
print(joined["days_before_request"].value_counts().sort_index())

print("\nRadii counts:")
print(joined["buffer_radius_m"].value_counts(dropna=False))