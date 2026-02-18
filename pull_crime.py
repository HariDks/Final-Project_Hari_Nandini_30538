#!/usr/bin/env python

import os
import time
import pandas as pd
from sodapy import Socrata
  
client = Socrata(
    "data.cityofchicago.org",
    "Lg7zqHMbACUvgMjA2CcB3iPyH",
    timeout=120
)

dataset_id = "ijzp-q8t2"
select_cols = "id,date,year,primary_type,latitude,longitude,community_area,beat,district,ward"
where_clause = "date between '2011-01-01T00:00:00.000' and '2018-12-31T23:59:59.999'"

limit = 10000
offset = 0
rows = []

while True:
    batch = client.get(dataset_id, select=select_cols, where=where_clause, limit=limit, offset=offset)
    if not batch:
        break
    rows.extend(batch)
    offset += limit
    print("rows:", len(rows))
    time.sleep(0.2)

df = pd.DataFrame.from_records(rows)
df.to_csv("crimes_2011_2018.csv", index=False)
print(df.shape)

