#!/usr/bin/env python3
import argparse
import os
from datetime import datetime

import geojson
import matplotlib.pyplot as plt
import netCDF4 as netcdf
import numpy as np
from mpl_toolkits.basemap import Basemap

from colors import chmi_colors, colors, front_names


def extract(date, variable_name, level_id):
    file = f"datasets/ecmwf/netcdf/era5_{date.year}.nc"
    if os.path.isfile(file):
        data = netcdf.Dataset(file, mode="r")  # read the data

        times = data.variables["time"]
        levels = data.variables["level"][:]
        lat = data.variables["latitude"][:]
        lon = data.variables["longitude"][:]
        variables = dict(data.variables)

        if 0:
            print(f"Available data: {variables}")
            print(f"Pressure levels: {levels}")

        if variable_name in variables:
            variable = data.variables[variable_name]
        else:
            print("Key not found, trying _001 variant.")
            variable = data.variables[variable_name + "_0001"]

        # print('units = %s, values = %s' % (times.units, times[:]))

        dates = netcdf.num2date(times[:], times.units)
        # print([date.strftime('%Y-%m-%d %H:%M:%S') for date in dates[:10]])

        ntime = netcdf.date2index(date, times, select="nearest")
        print(f"choosed #{ntime} ({dates[ntime]}) {levels[level_id]} hPa")
        print(f"unit: {variable.units}")

        final = variable[ntime, level_id, :, :]
        data.close()
        return (final, lat, lon)
    else:
        print(f"Dataset file not found: {file}")
        return ([], [], [])


def norm(values):
    # to floats 0-1
    min_val = np.min(values)
    max_val = np.max(values)
    normalized_data = (values - min_val) / (max_val - min_val)
    return normalized_data

"""
def create_product(target_date):
    t_a, lat, lon = extract(target_date, "t", 3)
    t_b, lat, lon = extract(target_date, "t", 4)
    # w, lat, lon = extract(target_date, "w", 2)
    cc, lat, lon = extract(target_date, "cc", 3)
    q, lat, lon = extract(target_date, "q", 3)
    # d, lat, lon = extract(target_date, "d", 3)
    clwc, lat, lon = extract(target_date, "clwc", 3)
    r = abs(t_a - t_b) * (clwc * 0.1)  # 850 hPa temp
    g = q
    b = cc

    r = norm(r)
    g = norm(g)
    b = norm(b)
    return (np.stack((r, g, b), axis=-1), lat, lon)
"""
def calculate_frontogenesis(temperature):
    dx = 1.0  # Assume grid spacing in x direction
    dy = 1.0  # Assume grid spacing in y direction
    
    dTx_dy = np.gradient(temperature, axis=0) / dy  # Partial derivative of temperature gradient in y direction
    dTy_dx = np.gradient(temperature, axis=1) / dx  # Partial derivative of temperature gradient in x direction
    
    d2Tx_dy2 = np.gradient(dTx_dy, axis=0) / dy  # Second partial derivative of temperature gradient in y direction
    d2Ty_dx2 = np.gradient(dTy_dx, axis=1) / dx  # Second partial derivative of temperature gradient in x direction
    
    frontogenesis = -d2Tx_dy2 + d2Ty_dx2
    
    return frontogenesis

def calculate_potential_vorticity(u, v, temperature):
    # Calculate potential vorticity
    # This is a simplified calculation assuming a constant potential temperature gradient
    dx = 1.0  # Assume grid spacing in x direction
    dy = 1.0  # Assume grid spacing in y direction
    dT_dy = np.gradient(temperature, axis=0) / dy
    dT_dx = np.gradient(temperature, axis=1) / dx
    dU_dx = np.gradient(u, axis=1) / dx
    dV_dy = np.gradient(v, axis=0) / dy
    PV = (1 / temperature) * (dV_dy - dU_dx) + (9.8 / temperature) * (dT_dx * v - dT_dy * u)
    return PV

def create_product(target_date):
    u_wind, lat, lon = extract(target_date, "u", 0)
    v_wind, lat, lon = extract(target_date, "v", 0)
    wind_speed = np.sqrt(u_wind**2+v_wind**2)
    
    #hum_1, lat, lon = extract(target_date, "q", 1)
    hum_2, lat, lon = extract(target_date, "q", 2)
    #hum_3, lat, lon = extract(target_date, "q", 3)
    #hum_4, lat, lon = extract(target_date, "q", 4)
    #abs_humidity = np.maximum.reduce([hum_2, hum_3, hum_4])
    
    #clouds, lat, lon = extract(target_date, "cc", 0)
    #water_content, lat, lon = extract(target_date, "clwc", 5)
    #cloud_composite = clouds+(water_content*2)
    
    temp1, lat, lon = extract(target_date, "t", 1)
    #temp2, lat, lon = extract(target_date, "t", 2)
    #temp3, lat, lon = extract(target_date, "t", 3)
    temp4, lat, lon = extract(target_date, "t", 4)
    #temp5, lat, lon = extract(target_date, "t", 5)
    #temperature_3d = np.stack((temp1, temp2, temp3, temp4, temp5), axis=-1)
    #thermal_gradient = np.gradient(temperature_3d, axis=2)
    
    frontogenesis = calculate_frontogenesis(hum_2) # best subjective choice
    pv = calculate_potential_vorticity(u_wind, v_wind, temp1)
    #temperature = temp4
    #temperature_median = np.median(temperature)
    #temperature[(temperature > temperature_median-1.5) & (temperature < temperature_median+1.5)] = 250
    
    r = pv
    g = (1-frontogenesis)*wind_speed #1-np.amax(thermal_gradient, axis=2)
    b = temp4

    r = norm(r)
    g = norm(g)
    b = norm(b)
    return (np.stack((r, g, b), axis=-1), lat, lon)


def create_product_v1(target_date):
    extracted, lat, lon = extract(target_date, "t", 3)
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
