from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
data_dir = Path("/Users/haridharshinik.s/Final-Project_Hari_Nandini_30538/data")
during_file = data_dir / "streetlight_crime_events.geojson"
out_png6 = data_dir / "crime_type_composition_during_by_radius.png"

# -----------------------
# Load DURING (crime-level)
# -----------------------
during = gpd.read_file(during_file)

needed = {"request_id", "buffer_radius_m", "id", "primary_type"}
missing = needed - set(during.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

during["buffer_radius_m"] = pd.to_numeric(during["buffer_radius_m"], errors="coerce")
during = during.dropna(subset=["buffer_radius_m", "primary_type", "id"]).copy()

# -----------------------
# Count unique crimes by (radius, type)
# -----------------------
counts = (
    during
    .groupby(["buffer_radius_m", "primary_type"], as_index=False)
    .agg(n_crimes=("id", "nunique"))
)

# Keep top K types overall for readability
K = 8
top_types = (
    counts.groupby("primary_type")["n_crimes"].sum()
    .sort_values(ascending=False)
    .head(K)
    .index
)

counts["type_plot"] = counts["primary_type"].where(counts["primary_type"].isin(top_types), "OTHER")

counts2 = (
    counts
    .groupby(["buffer_radius_m", "type_plot"], as_index=False)
    .agg(n_crimes=("n_crimes", "sum"))
)

# Convert to shares within each radius
counts2["share"] = counts2["n_crimes"] / counts2.groupby("buffer_radius_m")["n_crimes"].transform("sum")

# Pivot to wide for stacked bars
pivot = counts2.pivot(index="buffer_radius_m", columns="type_plot", values="share").fillna(0)

# Sort radii
pivot = pivot.loc[sorted(pivot.index)]

# -----------------------
# Plot: stacked bar chart of shares
# -----------------------
plt.figure(figsize=(10, 5))
bottom = None

x_labels = [f"{int(r)} m" for r in pivot.index]

for col in pivot.columns:
    vals = pivot[col].values
    if bottom is None:
        plt.bar(x_labels, vals, label=col)
        bottom = vals
    else:
        plt.bar(x_labels, vals, bottom=bottom, label=col)
        bottom = bottom + vals

plt.xlabel("Buffer radius")
plt.ylabel("Share of crimes (during service window)")
plt.title("Crime type composition during service window (top types grouped)")
plt.legend(title="Crime type", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(out_png6, dpi=200)
plt.show()

print(f"Saved: {out_png6}")