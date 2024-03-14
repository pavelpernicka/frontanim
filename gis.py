#!/usr/bin/env python3
import matplotlib.pyplot as plt
import rasterio
from osgeo import gdal

from coords_convert import CoordsConverter
from grib_reader import *

target_date = datetime(2017, 1, 1, 12)
geotiff_path = "datasets/generated_overlay/tif/01_01_2017.tif"
geojson_path = "datasets/generated_overlay/geojson/01_01_2017.geojson"


# Load GeoTIFF
with rasterio.open(geotiff_path) as src:
    data = src.read()
    crs = src.crs
    transform = src.transform
    extent = src.bounds

coord_converter = CoordsConverter()
gdal_ds = gdal.Open(geotiff_path)
xoff, a, b, yoff, d, e = gdal_ds.GetGeoTransform()


# Function to read GeoTIFF file and get its geospatial information
def read_geotiff(file_path):
    dataset = gdal.Open(file_path)
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return dataset, geotransform, projection


# Function to read NetCDF file and extract latitude, longitude, and data
def read_netcdf():
    extracted, lat, lon = extract(target_date, "t", 0)
    return lat, lon, extracted


# Function to find nearest latitude and longitude indices in the NetCDF dataset
def find_nearest_indices(lat, lon, target_lat, target_lon):
    lat_idx = np.abs(lat - target_lat).argmin()
    lon_idx = np.abs(lon - target_lon).argmin()
    return lat_idx, lon_idx


def create_and_display_image_from_netcdf(geotiff_path):
    # Read GeoTIFF
    geotiff_dataset, geotransform, _ = read_geotiff(geotiff_path)
    rows, cols = geotiff_dataset.RasterYSize, geotiff_dataset.RasterXSize

    # Read NetCDF
    lat, lon, data = read_netcdf()

    # Create new image
    output_image = np.zeros((rows, cols))

    # Iterate through each pixel in the GeoTIFF
    for i in range(rows):
        for j in range(cols):
            # Get geographic coordinates of pixel
            x = geotransform[0] + j * geotransform[1] + i * geotransform[2]
            y = geotransform[3] + j * geotransform[4] + i * geotransform[5]

            # Find nearest latitude and longitude indices
            lat_idx, lon_idx = find_nearest_indices(lat, lon, y, x)

            # Get data value corresponding to nearest latitude and longitude
            output_image[i, j] = data[lat_idx, lon_idx]

    # Display the image
    plt.imshow(output_image)
    plt.colorbar()
    plt.title("Image from NetCDF Data")
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    plt.show()


# Example usage
create_and_display_image_from_netcdf(geotiff_path)
