#!/usr/bin/env python3
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from osgeo import gdal
from rasterio.plot import show

from coords_convert import CoordsConverter
from grib_reader import *

target_date = datetime(2017, 1, 1, 12)
geotiff_path = "datasets/generated_overlay/tif/01_01_2017.tif"
geojson_path = "datasets/generated_overlay/geojson/01_01_2017.geojson"
extracted, lat, lon = extract(target_date, "t", 0)

# Load GeoTIFF
with rasterio.open(geotiff_path) as src:
    data = src.read()
    crs = src.crs
    transform = src.transform
    extent = src.bounds

coord_converter = CoordsConverter()
gdal_ds = gdal.Open(geotiff_path)
xoff, a, b, yoff, d, e = gdal_ds.GetGeoTransform()


def lat_lon(point):
    x, y = point
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    converted = coord_converter.coords((xp, yp))
    lat, lon = converted
    return (lon, lat)


print(lat)
from_netcdf = np.zeros((data.shape[1], data.shape[2]))
for x in range(data.shape[1]):
    for y in range(data.shape[2]):
        lat_d, lon_d = lat_lon((x, y))
        idx = np.abs(lat - lat_d).argmin()
        idy = np.abs(lon - lon_d).argmin()
        datapoint = extracted[idx, idy]
        print(
            f"Cartesian: [{x}, {y}] => geogrphic: [{lat_d}, {lon_d}]: [{idx}, {idy}]={datapoint}"
        )
        from_netcdf[idx, idy] = datapoint


# Load GeoJSON
gdf = gpd.read_file(geojson_path)

# Reproject GeoJSON to match GeoTIFF CRS
gdf_reprojected = gdf.to_crs(crs)

# Create plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot GeoTIFF
# show(data, transform=transform, ax=ax, extent=extent, cmap='viridis')
# print(extent)

# Plot NetCDF
show(from_netcdf)

# Plot GeoJSON
# ygdf_reprojected.plot(ax=ax, facecolor='none', edgecolor='red')


# Set plot title
plt.title("Aligned GeoTIFF and GeoJSON")

# Show plot
plt.show()
