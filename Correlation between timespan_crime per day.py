from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
data_dir = Path("/Users/haridharshinik.s/Final-Project_Hari_Nandini_30538/data")
data_dir.mkdir(parents=True, exist_ok=True)

input_csv = data_dir / "streetlight_level_crime_during_service.csv"
out_png1 = data_dir / "scatter_days_open_vs_crimes_per_day.png"

# -----------------------
# Load streetlight-level dataset
# -----------------------
df = pd.read_csv(input_csv)

needed = {"days_open", "crimes_per_day", "buffer_radius_m", "request_id"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -----------------------
# Clean
# -----------------------
df["days_open"] = pd.to_numeric(df["days_open"], errors="coerce")
df["crimes_per_day"] = pd.to_numeric(df["crimes_per_day"], errors="coerce")
df["buffer_radius_m"] = pd.to_numeric(df["buffer_radius_m"], errors="coerce")
df["request_id"] = df["request_id"].astype(str)

df = df.dropna(subset=["days_open", "crimes_per_day", "buffer_radius_m"]).copy()
df = df[(df["days_open"] > 0) & (df["crimes_per_day"] >= 0)].copy()

# -----------------------
# Optional winsorization / trimming of extreme durations
# Option A (winsorize at 99th percentile):
# -----------------------
upper = df["days_open"].quantile(0.99)
df["days_open_win"] = df["days_open"].clip(upper=upper)

# If you prefer trimming instead of winsorizing, comment the 2 lines above
# and uncomment this:
# df = df[df["days_open"] <= 365].copy()
# df["days_open_win"] = df["days_open"]

# -----------------------
# Bin durations: 0–7, 7–30, 30–90, 90+
# -----------------------
bins = [0, 7, 30, 90, np.inf]
labels = ["0–7", "7–30", "30–90", "90+"]

df["duration_bin"] = pd.cut(
    df["days_open_win"],
    bins=bins,
    labels=labels,
    right=False
)

df = df.dropna(subset=["duration_bin"]).copy()

# -----------------------
# Aggregate: avg crimes_per_day by (radius x bin)
# -----------------------
binned = (
    df
    .groupby(["buffer_radius_m", "duration_bin"], as_index=False)
    .agg(
        avg_crimes_per_day=("crimes_per_day", "mean"),
        n_requests=("request_id", "nunique"),
    )
)

# Make sure bins are in the intended order
binned["duration_bin"] = pd.Categorical(binned["duration_bin"], categories=labels, ordered=True)
binned = binned.sort_values(["buffer_radius_m", "duration_bin"])

# -----------------------
# Plot: line plot with markers (one line per radius)
# -----------------------
plt.figure(figsize=(8, 5))

for radius in sorted(binned["buffer_radius_m"].unique()):
    temp = binned[binned["buffer_radius_m"] == radius]
    plt.plot(
        temp["duration_bin"],
        temp["avg_crimes_per_day"],
        marker="o",
        label=f"{int(radius)} m"
    )

plt.xlabel("Request duration bin (days)")
plt.ylabel("Average crimes per day during service window")
plt.title("Crime intensity by request duration bin (winsorized days_open)")
plt.legend(title="Radius")
plt.tight_layout()
plt.savefig(out_png1, dpi=200)
plt.show()

print(f"Saved: {out_png1}")

# -----------------------
# Optional: print sample sizes per bin for context
# -----------------------
print("\nN requests per (radius, bin):")
print(binned.pivot_table(index="duration_bin", columns="buffer_radius_m", values="n_requests", fill_value=0))