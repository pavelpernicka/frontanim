#!/usr/bin/env python3
"""
TODO: command line interface
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
import pandas as pd
import datetime
from grib_reader import norm, lognorm
from colors import chmi_colors, colors, front_names
import geojson

# Load data
target_date = datetime.datetime(2020, 5, 18, 12)
f = open("datasets/generated_nooverlay/geojson/18_05_2020.geojson")
ds = xr.open_dataset('datasets/ecmwf/netcdf/era5_2020.nc').metpy.parse_cf()
geojson_data = geojson.load(f)

lats = ds.latitude.sel().values
lons = ds.longitude.sel().values
print(ds)
frontogenesis_level = 850 * units.hPa
jetstream_level = 300 * units.hPa
pres = ds['level'].values[:] * units('hPa')
t = ds['t'][:,:,:].metpy.sel(method='nearest', time=target_date)
u = ds['v'][:,:,:].metpy.sel(method='nearest', time=target_date)
v = ds['u'][:,:,:].metpy.sel(method='nearest', time=target_date)
q_850 = ds['q'][:,:,:].metpy.sel(level=frontogenesis_level, method='nearest', time=target_date).metpy.unit_array.squeeze()

datacrs = ccrs.PlateCarree()
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


humidity_gradient_raw = mpcalc.gradient(q_850, coordinates=(lats, lons))
humidity_gradient = np.sqrt(humidity_gradient_raw[0]**2 + humidity_gradient_raw[1]**2)

dewpoint = mpcalc.dewpoint_from_specific_humidity(frontogenesis_level, q_850)
eqpt = mpcalc.equivalent_potential_temperature(frontogenesis_level, t_850, dewpoint)
temperature_gradient_raw = mpcalc.gradient(eqpt, coordinates=(lats, lons))
temperature_gradient = np.sqrt(temperature_gradient_raw[0]**2 + temperature_gradient_raw[1]**2)
#temperature_gradient = temperature_gradient * np.where(temperature_gradient.m > 1e-18, 1, 0)

t_2 = np.stack((temperature_gradient_raw[0] / temperature_gradient, temperature_gradient_raw[1] / temperature_gradient), axis=-1)
t_1 = mpcalc.gradient(-potential_temperature_850, coordinates=(lats, lons))
tfp = t_1[1] * t_2[:,:,1] + t_1[0] * t_2[:,:,0] #Thermal front parameter

potential_vorticity = mpcalc.potential_vorticity_baroclinic(potential_temperature, pres[:, None, None], u, v, dx[None, :, :], dy[None, :, :], lats[None, :, None] * units('degrees')).sel(level=300 * units("hPa"), method='nearest')
rel_vorticity = mpcalc.vorticity(u_850, v_850, dx=dx, dy=dy)* 1e5
frontogenesis_850 = mpcalc.frontogenesis(potential_temperature_850, u_850, v_850, dx, dy)*1000*100*3600*3

u_qvect, v_qvect = mpcalc.q_vector(u_850, v_850, t_850, frontogenesis_level, dx, dy)
q_divergence = -2*mpcalc.divergence(u_qvect, v_qvect, dx=dx, dy=dy)

t_advection = mpcalc.advection(t_850, u_850, v_850, dx=dx, dy=dy)
hum_mask = (q_850 > 0.0025)
lengthq = np.sqrt(u_qvect**2 + v_qvect**2)
# Configure map plot
mapcrs = ccrs.Mercator()


fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-80, 60, 80, 30], ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

plt.title('Q-vectors 850 hPa', loc='left')
plt.title('Valid Time: {}'.format(target_date), loc='right')

"""
csf = ax.contourf(lons, lats, temperature_gradient, np.arange(0, 35, 1), transform=datacrs, cmap="Reds")
cb = plt.colorbar(csf, orientation='vertical', pad=0, aspect=50, extendrect=True)
"""

"""
csf = ax.contourf(lons, lats, rel_vorticity, np.arange(-30, 30, 2), transform=datacrs, cmap="seismic")
cb = plt.colorbar(csf, orientation='vertical', pad=0, aspect=50, extendrect=True)
"""

"""
csf = ax.contourf(lons, lats, t_advection, transform=datacrs, cmap="inferno")
cb = plt.colorbar(csf, orientation='vertical', pad=0, aspect=50, extendrect=True)
plt.clabel(csf, fmt='%d')
"""

"""
#, np.arange(0, 15, 0.5)
csf = ax.contourf(lons, lats, tfp, np.arange(-30, 30, 2), transform=datacrs, cmap="seismic")
cb = plt.colorbar(csf, orientation='vertical', pad=0, aspect=50, extendrect=True)
"""

"""
csf = ax.contourf(lons, lats, temperature_celsius_850, transform=datacrs, cmap="inferno")
cb = plt.colorbar(csf, orientation='vertical', pad=0, aspect=50, extendrect=True)
"""

"""
csf = ax.contourf(lons, lats, wind_speed_300, transform=datacrs, cmap="Reds")
cb = plt.colorbar(csf, orientation='vertical', pad=0, aspect=50, extendrect=True)
"""

"""
# Plot 850-hPa Frontogenesis
frontog, lons_cyclic = add_cyclic_point(frontogenesis_850*hum_mask, coord=lons)
cf = ax.contourf(lons_cyclic, lats, frontog, np.arange(-8, 8, 0.5),
                 cmap="seismic", extend='both', transform=datacrs, alpha=0.8)
cb = plt.colorbar(cf, orientation='vertical', pad=0, aspect=50, extendrect=True)
cb.set_label('Frontogenesis K / 100 km / 3 h')
"""


# Plot 850-hPa Q-Vector Divergence and scale
clevs_850_tmpc = np.arange(-40, 41, 2)
clevs_qdiv = list(range(-30, -4, 5))+list(range(5, 31, 5))
cf = ax.contourf(lons, lats, q_divergence*1e17*hum_mask, clevs_qdiv, cmap="BrBG",
                 extend='both', transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0, aspect=50, extendrect=True,
                  ticks=clevs_qdiv)
cb.set_label('Q-Vector Divergence (*10$^{17}$ m s$^{-1}$ kg$^{-1}$)')


# Plot 850-hPa Q-vectors, scale to get nice sized arrows
wind_slice = (slice(None, None, 5), slice(None, None, 5))
ax.quiver(lons[wind_slice[0]], lats[wind_slice[1]],
          u_qvect[wind_slice].m,
          v_qvect[wind_slice].m,
          pivot='mid', color='black',
          scale=7e-10,
          transform=datacrs)

for feature in geojson_data["features"]:
    front_type_id = front_names.index(feature["properties"]["front_type"])
    color = tuple(colors[front_type_id] / 255)
    if feature["geometry"]["type"] == "LineString":
        coordinates = feature["geometry"]["coordinates"]
        lon = [point[0] for point in coordinates]
        lat = [point[1] for point in coordinates]
        x, y = lon, lat
        ax.plot(x, y, color=color, linewidth=3, transform=datacrs)

plt.show()