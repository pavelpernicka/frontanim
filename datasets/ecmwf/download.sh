#!/bin/bash
directory="/home/pavel/frontanim/datasets/ecmwf"s
#curl https://download-0000-clone.copernicus-climate.eu/cache-compute-0000/cache/data2/adaptor.mars.internal-1709036916.2019613-23712-3-e012874a-01cb-4cb7-9230-775a382da566.grib -O "$directory/grib/era5_2024.grib"
#curl -o "$directory/grib/era5_2023.grib" https://download-0019.copernicus-climate.eu/cache-compute-0019/cache/data0/adaptor.mars.internal-1709037922.591747-23609-2-e903a6b4-34b0-4beb-8426-96391acf6ec3.grib
#curl -o "$directory/grib/era5_2022.grib" https://download-0008-clone.copernicus-climate.eu/cache-compute-0008/cache/data5/adaptor.mars.internal-1709040973.3143542-2187-19-860505b7-e4ad-4401-acfa-6cd0e4f34069.grib
#curl -o "$directory/grib/era5_2021.grib" https://download-0005-clone.copernicus-climate.eu/cache-compute-0005/cache/data9/adaptor.mars.internal-1709043811.6595252-24852-15-80d9b783-b3de-4d92-939c-9a373568ad4f.grib
#curl -o "$directory/grib/era5_2020.grib" https://download-0001-clone.copernicus-climate.eu/cache-compute-0001/cache/data4/adaptor.mars.internal-1709046107.4218354-1174-2-5414cb14-aad3-49a1-a739-cae499a288e7.grib
#curl -o "$directory/grib/era5_2019.grib" https://download-0017.copernicus-climate.eu/cache-compute-0017/cache/data5/adaptor.mars.internal-1709050152.1245904-5859-6-a5734291-3bfe-42a0-be17-0c817509c70e.grib
#curl -o "$directory/grib/era5_2018.grib" https://download-0019.copernicus-climate.eu/cache-compute-0019/cache/data6/adaptor.mars.internal-1709053244.2153869-25386-8-44959cca-d069-4042-b1cf-ef4c0a6ed875.grib
#curl -o "$directory/grib/era5_2017.grib" https://download-0004-clone.copernicus-climate.eu/cache-compute-0004/cache/data2/adaptor.mars.internal-1709060337.150452-3318-1-c859c671-1fdd-435e-8d7b-a2b5c504721c.grib

grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2024.nc" "$directory/grib/era5_2024.grib"
grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2023.nc" "$directory/grib/era5_2023.grib"
grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2022.nc" "$directory/grib/era5_2022.grib"
grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2021.nc" "$directory/grib/era5_2021.grib"
grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2020.nc" "$directory/grib/era5_2020.grib"
grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2019.nc" "$directory/grib/era5_2019.grib"
grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2018.nc" "$directory/grib/era5_2018.grib"
grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2017.nc" "$directory/grib/era5_2017.grib"
grib_to_netcdf -k 4 -o "$directory/netcdf/era5_2014.nc" "$directory/grib/era5_2014.grib"