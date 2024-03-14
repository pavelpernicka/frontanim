#!/usr/bin/env python3
import os
import sys
from collections import Counter
from datetime import datetime

import cdsapi
import PIL.Image

from imgformat import *

years = []
months = []
days = []
hours = []

folder = "datasets/generated_overlay/png"
for filename in os.listdir(folder):
    img_for_meta = PIL.Image.open(folder + "/" + filename)
    meta = get_metadata(img_for_meta)
    img_for_meta.close()
    date = datetime.fromtimestamp(meta["datetime"])
    y = date.strftime("%Y")
    m = date.strftime("%m")
    d = date.strftime("%d")
    h = date.strftime("%H:00")
    years.append(y)
    months.append(m)
    days.append(d)
    hours.append(h)
    # print(date.strftime("%Y-%m-%dT%H:00"))


def count(inp):
    element_counts = Counter(inp)
    return [(element, count) for element, count in element_counts.items()]


print(count(years))
print(count(months))
print(count(days))
print(count(hours))

c = cdsapi.Client()

# for year in list(set(years)):
for year in [sys.argv[1]]:
    print(f"Downloading year:  {year}...")
    fn = "datasets/ecmwf/era5_" + year + ".grib"
    if os.path.exists(fn):
        print("File already exists, skipping")
        break
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "year": year,
            "month": list(set(months)),
            "day": list(set(days)),
            "time": list(set(hours)),
            "area": [
                80,
                -80,
                30,
                60,
            ],
            "pressure_level": [
                "450",
                "550",
                "650",
                "750",
                "850",
                "950",
                "1000",
            ],
            "variable": [
                "divergence",
                "fraction_of_cloud_cover",
                "specific_cloud_liquid_water_content",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
            ],
        },
        fn,
    )
