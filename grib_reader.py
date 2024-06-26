#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
import metpy.calc as mpcalc
from metpy.units import units
import geojson
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap
from colors import chmi_colors, colors, front_names

def norm(values):
    # to floats 0-1
    min_val = np.min(values)
    max_val = np.max(values)
    normalized_data = (values - min_val) / (max_val - min_val)
    return normalized_data
    
def lognorm(array):
    array_shifted = array - np.min(array) + 1e-9
    log_array = np.log10(np.abs(array_shifted))
    normalized_array = (log_array - np.min(log_array)) / (np.max(log_array) - np.min(log_array))
    return normalized_array

def custom_normalize(arr):
    # Replace NaN values with zeros
    arr = np.nan_to_num(arr)
    
    # Find the minimum and maximum values, excluding NaNs
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    # Handle the case where min and max are the same (i.e., all values are zeros or NaNs)
    if min_val == max_val:
        # If all values are zeros or NaNs, return the array itself
        return arr
    
    # Normalize the array to the range [0, 1]
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr


def create_product(target_date):
    file = f"datasets/ecmwf/netcdf/era5_{target_date.year}.nc"
    ds = xr.open_dataset(file).metpy.parse_cf()
    
    lats = ds.latitude.sel().values
    lons = ds.longitude.sel().values
    print(ds)
    
    frontogenesis_level = 850 * units.hPa
    jetstream_level = 300 * units.hPa
    
    pres = ds['level'].values[:] * units('hPa')
    t = ds['t'].metpy.sel(method='nearest', time=target_date)
    u = ds['v'].metpy.sel(method='nearest', time=target_date)
    v = ds['u'].metpy.sel(method='nearest', time=target_date)
    q_850 = ds['q'].metpy.sel(level=frontogenesis_level, method='nearest', time=target_date).metpy.unit_array.squeeze()

    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    potential_temperature = mpcalc.potential_temperature(pres[:, None, None] , t)
    u_300 = u.sel(level=jetstream_level, method='nearest').metpy.unit_array.squeeze()
    v_300 = v.sel(level=jetstream_level, method='nearest').metpy.unit_array.squeeze()
    u_850 = u.sel(level=frontogenesis_level, method='nearest').metpy.unit_array.squeeze()
    v_850 = v.sel(level=frontogenesis_level, method='nearest').metpy.unit_array.squeeze()
    t_850 = t.sel(level=frontogenesis_level, method='nearest').metpy.unit_array.squeeze()
    potential_temperature_850 = mpcalc.potential_temperature(850 * units("hPa") , t_850)
    temperature_celsius_850 = t_850.to('degC')
    wind_speed_300 = mpcalc.wind_speed(u_300, v_300)

    #humidity_gradient_raw = mpcalc.gradient(q_850, coordinates=(lats, lons))
    #humidity_gradient = np.sqrt(humidity_gradient_raw[0]**2 + humidity_gradient_raw[1]**2)
    hum_mask = (q_850.m > 0.0025)

    #temperature_gradient_raw = mpcalc.gradient(q_850, coordinates=(lats, lons))
    #temperature_gradient = np.sqrt(temperature_gradient_raw[0]**2 + temperature_gradient_raw[1]**2)

    potential_vorticity = mpcalc.potential_vorticity_baroclinic(potential_temperature, pres[:, None, None], u, v, dx[None, :, :], dy[None, :, :], lats[None, :, None] * units('degrees')).sel(level=300 * units("hPa"), method='nearest')
    frontogenesis_850 = mpcalc.frontogenesis(potential_temperature_850, u_850, v_850, dx, dy)*1000*100*3600*3

    #u_qvect, v_qvect = mpcalc.q_vector(u_850, v_850, t_850, frontogenesis_level, dx, dy)
    #lengthq = np.sqrt(u_qvect**2 + v_qvect**2)
    
    #t_advection = mpcalc.advection(t_850, u_850, v_850, dx=dx, dy=dy)
    a = abs(frontogenesis_850.m*hum_mask)
    a[a>5] = 5
    r = a + np.array(potential_vorticity)
    g = np.array(potential_temperature_850.m)
    b = wind_speed_300.m
    
    r = custom_normalize(r)
    g = custom_normalize(g)
    b = custom_normalize(b)
    return (np.stack((r, g, b), axis=-1), lats, lons)


def create_product_v1(target_date):
    extracted, lat, lon = extract(target_date, "t", 850)
    return (extracted, lat, lon)


def mkimg(
    front_file="",
    enable_map=True,
    enable_data=True,
    enable_fronts=True,
    enable_header=False,
    save_as="",
    custom_creator=create_product,
    custom_text="FPP",
    custom_datetime=None
):
    if(front_file!=""):
        try:
            f = open(front_file)
            geojson_data = geojson.load(f)
        except:
            print(f"Cannot open {front_file}")
            return False
        meta = geojson_data["properties"]
        target_date = datetime.fromtimestamp(meta["datetime"])
    else:
        target_date = custom_datetime
    print(f"Target datetime: {target_date}")
    # target_date = datetime(2021, 5, 17, 12)
    limits = [-55.0, 30.0, 53.0, 80.0]
    # SN, NE, SN, NE, SN
    map = Basemap(
        projection="merc",
        llcrnrlon=limits[0],
        llcrnrlat=limits[1],
        urcrnrlon=limits[2],
        urcrnrlat=limits[3],
        resolution="i",
    )
    fig, ax = plt.subplots()

    # empty rectangle of map size to set plot extents if no map layers are enabled
    map.drawmapboundary(fill_color=None, color=None)

    if enable_map:
        map.drawcoastlines(ax=ax)
        map.drawcountries(ax=ax)
        map.drawlsmask(land_color="Linen", ocean_color="#CCFFFF", ax=ax)
        map.drawcounties(ax=ax)

    # print(lat, lon)
    if enable_data:
        final, lat, lon = custom_creator(target_date)
        # create meshgrid for plotting
        lons, lats = np.meshgrid(lon, lat)

        # convert coordinates to map projection
        x, y = map(lons, lats)

        # plot netcdf data
        contour = ax.pcolormesh(x, y, final, alpha=1)

    if enable_fronts:
        # plot GeoJSON features
        for feature in geojson_data["features"]:
            front_type_id = front_names.index(feature["properties"]["front_type"])
            color = tuple(colors[front_type_id] / 255)
            if feature["geometry"]["type"] == "LineString":
                coordinates = feature["geometry"]["coordinates"]
                lon = [point[0] for point in coordinates]
                lat = [point[1] for point in coordinates]
                x, y = map(lon, lat)
                ax.plot(x, y, color=color, linewidth=3)

    if enable_header:
        fdate = target_date.strftime("%Y-%m-%d %H UTC")
        plt.title(f"{custom_text} ({fdate})")

    # image = georaster.SingleBandRaster( "datasets/generated_nooverlay/tif/01_01_2017.tif", latlon=False)
    # plt.imshow(image.r, extent=image.extent, zorder=10, alpha=0.6)
    if save_as == "":
        plt.show()
    else:
        width = 512
        height = 512
        figure = plt.gcf()
        figure.set_size_inches(width / 100, height / 100)
        plt.axis("off")
        plt.gca().set_position([0, 0, 1, 1])
        plt.savefig(save_as, dpi=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create NWP model data image")
    parser.add_argument("--source", help="Input file path")
    parser.add_argument("--save", type=argparse.FileType('wb'), help="Save generated image as")
    parser.add_argument("--fronts", action="store_true", help="Show front overlay")
    parser.add_argument("--nodata", action="store_true", help="Do not show data layer")
    parser.add_argument("--map", action="store_true", help="Show map layer")
    parser.add_argument("--header", action="store_true", help="Show plot header")
    parser.add_argument("--date", help="Load from datasets by given datetime in format YYYY-MM-DD-HH")
    args = parser.parse_args()

    if args.save:
        target_file = args.save
        header = False
    else:
        target_file = ""
        header = True

    dt = None
    if args.date:
        dt = datetime.strptime(args.date, "%Y-%m-%d-%H")
        args.fronts = False
        args.source = ""
        
    mkimg(
        args.source,
        enable_header=header,
        enable_map=args.map,
        enable_fronts=args.fronts,
        save_as=target_file,
        enable_data=not args.nodata,
        custom_datetime=dt
    )
    # mkimg(args.source, enable_header=True) #human analyse
    # mkimg(args.source, enable_map=False, enable_fronts=False, save_as="bbb.png") #to ML - model data
    # mkimg(args.source, enable_data=False, enable_map=False, save_as="aaa.png") #to ML - front lines
