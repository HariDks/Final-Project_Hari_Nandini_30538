# join_streetlights_crime.py
# Spatiotemporal join:
# Crimes within streetlight buffers AND within service request time window

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# -----------------------
# Paths (USE THESE EXACT DIRECTORIES)
# -----------------------
project_dir = Path(__file__).parent
data_dir = project_dir / "data"

buffers_file = data_dir / "/Users/haridharshinik.s/Final-Project_Hari_Nandini_30538/data/streetlights_buffers.geojson"
crime_csv =  data_dir/"/Users/haridharshinik.s/Final-Project_Hari_Nandini_30538/crimes_2011_2018.csv"
output_file = data_dir / "streetlight_crime_events.geojson"

# -----------------------
# Load streetlight buffers
# -----------------------
buffers = gpd.read_file(buffers_file)

# Rename date columns if needed (handles your earlier wording: creating_date)
if "creation_date" not in buffers.columns and "creating_date" in buffers.columns:
    buffers = buffers.rename(columns={"creating_date": "creation_date"})
if "completion_date" not in buffers.columns and "completed_date" in buffers.columns:
    buffers = buffers.rename(columns={"completed_date": "completion_date"})

# Check required columns
needed_buf = {"creation_date", "completion_date", "buffer_radius_m", "geometry", "service_request_number"}
missing_buf = needed_buf - set(buffers.columns)
if missing_buf:
    raise ValueError(f"Missing buffer columns: {missing_buf}")

# Parse streetlight dates (works for 2011-01-01T00:00:00.000)
buffers["creation_date"] = pd.to_datetime(buffers["creation_date"], errors="coerce")
buffers["completion_date"] = pd.to_datetime(buffers["completion_date"], errors="coerce")

# Drop rows without creation date (can’t define a window)
buffers = buffers.dropna(subset=["creation_date"]).copy()

# Ensure CRS is projected (meters)
if buffers.crs is None:
    raise ValueError("Buffers have no CRS. Recreate buffers with a CRS.")

if buffers.crs.to_epsg() != 3435:
    buffers = buffers.to_crs(epsg=3435)

# Use service_request_number as the true request_id
buffers["request_id"] = buffers["service_request_number"].astype(str)

# Optional sanity: each request_id should appear once per radius
# (won't always be true if your source CSV itself contains duplicates)
# print(buffers.groupby(["request_id", "buffer_radius_m"]).size().sort_values(ascending=False).head())

# -----------------------
# Load crimes
# -----------------------
crime_df = pd.read_csv(crime_csv)

needed_crime = {"id", "date", "latitude", "longitude"}
missing_crime = needed_crime - set(crime_df.columns)
if missing_crime:
    raise ValueError(f"Missing crime columns: {missing_crime}")

# Parse crime date (works for 2011-01-01T00:00:00.000)
crime_df["crime_date"] = pd.to_datetime(crime_df["date"], errors="coerce")

# Drop bad rows
crime_df = crime_df.dropna(subset=["crime_date", "latitude", "longitude"]).copy()

# -----------------------
# Crimes → GeoDataFrame (WGS84) → Project to EPSG:3435
# -----------------------
crime_geom = [Point(xy) for xy in zip(crime_df["longitude"], crime_df["latitude"])]
crime_gdf = gpd.GeoDataFrame(crime_df, geometry=crime_geom, crs="EPSG:4326").to_crs(epsg=3435)

# -----------------------
# Spatial join: crimes within buffer polygons
# -----------------------
joined = gpd.sjoin(
    crime_gdf,
    buffers,
    how="inner",
    predicate="within",
)

print("After spatial join:", len(joined))

# -----------------------
# Temporal filter
# Keep crimes during the service window:
# creation_date ≤ crime_date ≤ completion_date
# If completion_date is missing → treat as "open request" (no upper bound)
# -----------------------
time_mask = (
    (joined["crime_date"] >= joined["creation_date"]) &
    (joined["completion_date"].isna() | (joined["crime_date"] <= joined["completion_date"]))
)

joined = joined[time_mask].copy()
print("After time filter:", len(joined))

# -----------------------
# Clean output
# -----------------------
cols = [
    # Crime info
    "id",
    "primary_type",
    "crime_date",
    "year",
    "community_area",
    "beat",
    "district",
    "ward",

    # Streetlight info
    "request_id",
    "service_request_number",
    "buffer_radius_m",
    "creation_date",
    "completion_date",
    "status",

    # Geometry (crime point)
    "geometry",
]
cols = [c for c in cols if c in joined.columns]
joined = joined[cols]

# -----------------------
# Save
# -----------------------
joined.to_file(output_file, driver="GeoJSON")
print(f"Saved joined file to: {output_file}")

# -----------------------
# Sanity checks
# -----------------------
print("\nPreview:")
print(joined.head(3))

print("\nRadii counts:")
print(joined["buffer_radius_m"].value_counts(dropna=False))

print("\nRequests (top 5):")
print(joined["request_id"].value_counts().head())

# Ensure datetime types (safe)
joined["crime_date"] = pd.to_datetime(joined["crime_date"], errors="coerce")
joined["creation_date"] = pd.to_datetime(joined["creation_date"], errors="coerce")
joined["completion_date"] = pd.to_datetime(joined["completion_date"], errors="coerce")

# Build mask fresh from *this* joined
time_mask = (
    (joined["crime_date"] >= joined["creation_date"]) &
    (joined["completion_date"].isna() | (joined["crime_date"] <= joined["completion_date"]))
)

# Filter by position, not index labels
joined = joined.loc[time_mask.to_numpy()].copy()

print("After time filter:", len(joined))

# -----------------------
# Streetlight-level aggregation
# one row per request_id x buffer_radius_m
# -----------------------
streetlight_level = (
    joined
    .groupby(["request_id", "buffer_radius_m"], as_index=False)
    .agg(
        crime_count=("id", "count"),
        unique_crimes=("id", "nunique"),
        creation_date=("creation_date", "first"),
        completion_date=("completion_date", "first"),
        status=("status", "first"),
    )
)

print("\nStreetlight-level preview:")
print(streetlight_level.head())

# Duration in days (can be NaN if completion_date missing)
streetlight_level["days_open"] = (
    streetlight_level["completion_date"] - streetlight_level["creation_date"]
).dt.total_seconds() / 86400.0

# Crime rate per day (only for requests with a completion_date)
streetlight_level["crimes_per_day"] = streetlight_level["unique_crimes"] / streetlight_level["days_open"]

streetlight_level = streetlight_level.dropna(subset=["completion_date"])

output_csv = data_dir / "streetlight_level_crime_during_service.csv"
streetlight_level.to_csv(output_csv, index=False)
print(f"Saved streetlight-level dataset to: {output_csv}")