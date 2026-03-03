# tracts_data_creation.py
# Spatial join: assign each streetlight-crime-buffer event row to a census tract.
# Output keeps the tract polygon geometry so downstream maps can colour by tract.

from pathlib import Path
import geopandas as gpd
import pandas as pd

# -----------------------
# Paths
# -----------------------
project_dir = Path(__file__).parent
data_dir = project_dir / "data"

events_file = data_dir / "streetlight_crime_events.geojson"
tracts_file = data_dir / "illinois_tract_income" / "illinois_tract_income.shp"
output_file = data_dir / "streetlight_crime_events_with_tracts.geojson"

# -----------------------
# Load events (crime point geometry, EPSG:3435 after sjoin.py)
# -----------------------
events = gpd.read_file(events_file)
print("Events loaded:", len(events))
print("Events CRS:", events.crs)

# -----------------------
# Load census tracts
# -----------------------
tracts = gpd.read_file(tracts_file)
print("Tracts loaded:", len(tracts))
print("Tracts CRS:", tracts.crs)

# -----------------------
# Align CRS — reproject tracts to match events
# -----------------------
if events.crs is None:
    raise ValueError("Events GeoDataFrame has no CRS. Re-run sjoin.py first.")

tracts = tracts.to_crs(events.crs)

# -----------------------
# Spatial join: each crime event point → containing tract polygon
# using left join so events with no matching tract (outside Illinois) are kept as NaN
# -----------------------
joined = gpd.sjoin(
    events,
    tracts[["GEOID", "NAME", "NAMELSAD", "ASQPE001", "ASQPM001", "geometry"]],
    how="left",
    predicate="within",
)

print("After tract join:", len(joined))
print("Events with tract match:", joined["GEOID"].notna().sum())
print("Events without tract match:", joined["GEOID"].isna().sum())

# -----------------------
# Drop the index_right column added by sjoin
# -----------------------
joined = joined.drop(columns=["index_right"], errors="ignore")

# Rename income columns for clarity
joined = joined.rename(columns={
    "ASQPE001": "median_income_estimate",
    "ASQPM001": "median_income_moe",
    "NAME":     "tract_name",
    "NAMELSAD": "tract_namelsad",
    "GEOID":    "tract_geoid",
})

# -----------------------
# Save — keep crime point geometry (not tract polygon)
# For tract polygon geometry, see the separate tract-level summary below
# -----------------------
joined.to_file(output_file, driver="GeoJSON")
print(f"Saved event-level file to: {output_file}")

# -----------------------
# Tract-level summary: aggregate events per tract, attach tract polygon geometry
# -----------------------
tract_agg = (
    joined.groupby("tract_geoid")
    .agg(
        total_crime_events=("id", "count"),
        unique_crimes=("id", "nunique"),
        crimes_during_outage=("crime_streetlight_outage", "sum"),
        median_income_estimate=("median_income_estimate", "first"),
        tract_name=("tract_name", "first"),
    )
    .reset_index()
)

# Re-attach tract polygon geometry
tract_polygons = tracts[["GEOID", "geometry"]].rename(columns={"GEOID": "tract_geoid"})
tract_summary = tract_polygons.merge(tract_agg, on="tract_geoid", how="inner")
tract_summary = gpd.GeoDataFrame(tract_summary, geometry="geometry", crs=events.crs)

tract_summary_file = data_dir / "tract_level_crime_summary.geojson"
tract_summary.to_file(tract_summary_file, driver="GeoJSON")
print(f"Saved tract-level summary to: {tract_summary_file}")

print("\nTract summary preview:")
print(tract_summary[["tract_geoid", "tract_name", "total_crime_events", "crimes_during_outage", "median_income_estimate"]].head())
