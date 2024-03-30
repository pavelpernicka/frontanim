#!/usr/bin/env python3
"""
Plot frontogenesis only
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

ds = xr.open_dataset('datasets/ecmwf/netcdf/era5_2020.nc').metpy.parse_cf()
print(ds.variables)

lats = ds.latitude.sel().values
lons = ds.longitude.sel().values

print(lats.shape)
print(lons.shape)

time = 553

level = 850 * units.hPa
tmpk_850 = ds['t'][time,:,:].metpy.sel(
    vertical=level, method='nearest').metpy.unit_array.squeeze()
uwnd_850 = ds['v'][time,:,:].metpy.sel(
    vertical=level, method='nearest').metpy.unit_array.squeeze()
vwnd_850 = ds['u'][time,:,:].metpy.sel(
    vertical=level, method='nearest').metpy.unit_array.squeeze()

print(type(tmpk_850))
print(uwnd_850.shape)
print(vwnd_850.shape)

tmpc_850 = tmpk_850.to('degC')
thta_850 = mpcalc.potential_temperature(level, tmpk_850)
vtime = ds.time.data[time].astype('datetime64[ms]').astype('O')
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

fronto_850 = mpcalc.frontogenesis(thta_850, uwnd_850, vwnd_850, dx, dy)
convert_to_per_100km_3h = 1000*100*3600*3


mapcrs = ccrs.Mercator()
datacrs = ccrs.PlateCarree()

fig = plt.figure(1, figsize=(14, 12))
ax = plt.subplot(111, projection=mapcrs)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Plot 850-hPa Frontogenesis
clevs_tmpc = np.arange(-40, 41, 2)

frontog, lons1 = add_cyclic_point(fronto_850*convert_to_per_100km_3h, coord=lons)
tpc, lons1 = add_cyclic_point(tmpc_850, coord=lons)

cf = ax.contourf(lons1, lats, frontog, np.arange(-8, 8.5, 0.5),
                 cmap=plt.cm.bwr, extend='both', transform=datacrs)
cb = plt.colorbar(cf, orientation='horizontal', pad=0, aspect=50, extendrect=True)
cb.set_label('Frontogenesis K / 100 km / 3 h')

# Plot titles
plt.title('Frontogenesis ERA5', loc='left')
plt.title('Valid Time: {}'.format(vtime), loc='right')

plt.show()