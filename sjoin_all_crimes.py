"""
sjoin_all_crimes.py
All crimes (2011–2018) spatially left-joined to 30m streetlight outage buffers.

Structure mirrors streetlight_crime_events.geojson but:
  - LEFT join  → every crime row is retained
  - No time-window filter → all crimes regardless of when relative to outage
  - Same derived columns (time_to_fix, days_from_outage_request,
    crime_streetlight_outage) where a match exists, NaN otherwise

Output: data/all_crimes_streetlight_joined.geojson
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

project_dir = Path(__file__).parent
data_dir    = project_dir / "data"
output_file = data_dir / "all_crimes_streetlight_joined.geojson"

# ---------------------------------------------------------------
# 1. Load & prepare streetlight buffers (30 m only)
# ---------------------------------------------------------------
print("Loading buffers...")
buffers = gpd.read_file(data_dir / "streetlights_buffers.geojson")

if "creation_date" not in buffers.columns and "creating_date" in buffers.columns:
    buffers = buffers.rename(columns={"creating_date": "creation_date"})
if "completion_date" not in buffers.columns and "completed_date" in buffers.columns:
    buffers = buffers.rename(columns={"completed_date": "completion_date"})

buffers["creation_date"]   = pd.to_datetime(buffers["creation_date"],   errors="coerce")
buffers["completion_date"] = pd.to_datetime(buffers["completion_date"], errors="coerce")
buffers = buffers.dropna(subset=["creation_date", "completion_date"]).copy()

if buffers.crs is None or buffers.crs.to_epsg() != 3435:
    buffers = buffers.to_crs(epsg=3435)

buffers["request_id"] = buffers["service_request_number"].astype(str)

# Keep 30 m buffers only — consistent with main analysis
buffers_30 = buffers[buffers["buffer_radius_m"] == 30].copy()
print(f"  30m buffer polygons: {len(buffers_30):,}")

buf_cols = [
    "request_id", "service_request_number", "buffer_radius_m",
    "creation_date", "completion_date", "status", "geometry",
]
buf_cols = [c for c in buf_cols if c in buffers_30.columns]
buffers_30 = buffers_30[buf_cols]

# ---------------------------------------------------------------
# 2. Load & prepare crimes
# ---------------------------------------------------------------
print("Loading crimes...")
crime_df = pd.read_csv(data_dir / "crimes_2011_2018.csv")
crime_df["crime_date"] = pd.to_datetime(crime_df["date"], errors="coerce")
crime_df = crime_df.dropna(subset=["crime_date", "latitude", "longitude"]).copy()
print(f"  Crime rows: {len(crime_df):,}")

crime_geom = [Point(xy) for xy in zip(crime_df["longitude"], crime_df["latitude"])]
crime_gdf  = gpd.GeoDataFrame(crime_df, geometry=crime_geom, crs="EPSG:4326").to_crs(epsg=3435)

# ---------------------------------------------------------------
# 3. Left spatial join — crimes → 30m buffers
# ---------------------------------------------------------------
print("Running spatial join (this may take a few minutes)...")
joined = gpd.sjoin(
    crime_gdf,
    buffers_30,
    how="left",
    predicate="within",
).drop(columns=["index_right"], errors="ignore")

print(f"  Rows after left join: {len(joined):,}")
print(f"  Crimes with a buffer match: {joined['request_id'].notna().sum():,}")
print(f"  Crimes with no buffer match: {joined['request_id'].isna().sum():,}")

# ---------------------------------------------------------------
# 4. Derived columns (only meaningful where a match exists)
# ---------------------------------------------------------------
joined["creation_date"]   = pd.to_datetime(joined["creation_date"],   errors="coerce")
joined["completion_date"] = pd.to_datetime(joined["completion_date"], errors="coerce")

# Days from outage report to crime (negative = crime before report)
joined["days_from_outage_request"] = (
    (joined["crime_date"] - joined["creation_date"]).dt.total_seconds() / 86400.0
)

# Duration of the outage in days
joined["time_to_fix"] = (
    (joined["completion_date"] - joined["creation_date"]).dt.total_seconds() / 86400.0
)

# 1 if crime occurred while outage was active, 0 if before/after, NaN if no match
def outage_flag(row):
    if pd.isna(row["days_from_outage_request"]) or pd.isna(row["time_to_fix"]):
        return float("nan")
    return int(0 <= row["days_from_outage_request"] <= row["time_to_fix"])

joined["crime_streetlight_outage"] = joined.apply(outage_flag, axis=1)

# ---------------------------------------------------------------
# 5. Select & order output columns
# ---------------------------------------------------------------
crime_cols = [
    "id", "primary_type", "crime_date", "year",
    "community_area", "beat", "district", "ward",
]
streetlight_cols = [
    "request_id", "service_request_number", "buffer_radius_m",
    "creation_date", "completion_date", "status",
]
derived_cols = [
    "crime_streetlight_outage", "time_to_fix", "days_from_outage_request",
]
out_cols = [c for c in crime_cols + streetlight_cols + derived_cols + ["geometry"]
            if c in joined.columns]
joined = joined[out_cols]

# ---------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------
print("Saving...")
joined.to_file(output_file, driver="GeoJSON")
print(f"Saved → {output_file}")
print(f"\nFinal shape: {joined.shape}")
print(joined.dtypes)
print("\nPreview:")
print(joined.head(3).to_string())
