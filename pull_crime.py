#!/usr/bin/env python

import pandas as pd
from sodapy import Socrata

# Connect to Chicago Data Portal
client = Socrata(
    "data.cityofchicago.org",
    "Lg7zqHMbACUvgMjA2CcB3iPyH"   # app token
)

all_results = []

limit = 50000   # max Socrata allows per request
offset = 0

while True:
    print(f"Requesting rows {offset} to {offset + limit}...")

    batch = client.get(
        "x2n5-8w5q",
        limit=limit,
        offset=offset
    )

    if len(batch) == 0:
        print("Done downloading.")
        break

    all_results.extend(batch)
    offset += limit

    print(f"Total rows so far: {len(all_results)}")

# Convert to DataFrame
df = pd.DataFrame.from_records(all_results)

# Save
df.to_csv("chicago_crime_all.csv", index=False)

print("Saved to chicago_crime_all.csv")
print("Final shape:", df.shape)