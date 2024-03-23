# Frontanim
Proof of concept for predicting atmospheric front placement from model data using a Generative Adversarial Network (GAN). The project includes useful tools to convert synoptic analysis images from the Czech Hydrometeorological Institute to GeoTIFFs and GeoJSONs for further usage.

The purpose of this project is to create synoptic maps of frontal lines from numerical weather prediction models. Currently, a pix2pix conditional GAN is implemented to transform images with NWP model data stored in RGB channels to another image of front lines. According to my experiments, the Hausdorff distance is used as the generator loss function.

## Installation
1. The project requires Python >= 3.6
2. Install the GDAL library: `sudo apt-get install libgdal-dev`
3. Install requirements: `pip install -r requirements.txt`

## Training Data
### NWP Model
The project works with model data in NetCDF format, so it is up to you to choose which NWP model to use. The most accessible way is to use ECMWF ERA5 reanalysis from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form). 
You need to download the following variables at these pressure levels: 450, 550, 650, 750, 850, 950, 1000.
On the download configuration page, do not forget to choose NetCDF format; otherwise, you will need to run `grib_to_netcdf -k 4 -o out.nc in.grib`. The `grib_to_netcdf` tool is from the `libeccodes-tools` package. For automated downloads, there is the `cdsapi` module for Python (see `downloader.py`), but due to long waiting times in the CDS queue, it is unusable. Be prepared to download about 16 GB of data per year (3 data series per day). Place the downloaded data into the `datasets/ecmwf/netcdf` folder named as `era5_YYYY.nc`.

### Synoptic Analysis Image
To generate the GeoJSON dataset, you need to obtain a few thousand synoptic analysis images from a meteorological institute. The current version of `extract_fronts.py` works with images made by the Czech Hydrometeorological Institute (https://www.chmi.cz/predpovedi/predpovedi-pocasi/evropa/synopticka-situace?l=en), but it can be easily customized for other options (you need to edit image sizing, masking boxes, and georeferencing points).
I personally have about 3000 images from 2015, but licenses prevent me from sharing them. However, I may be able to share generated GeoJSONs.

## Process from Images to Prediction
1. Download the required data and move them to the correct folders - NetCDFs from the model to `datasets/ecmwf/netcdf`, synoptic images to `datasets/synoptic_analysis_original`.
2. Run `./scripts/convert.sh <target folder>` to generate GeoTIFFs and GeoJSONs from all synoptic images.
3. Run `./scripts/mkdataset.sh` to generate training data pairs. You may need to change default folders directly in the script.
4. (Optional) Run `./scripts/augment.sh` to make the dataset larger by creating 90, 180, 270-degree rotated variants.
5. Start training: `screen ./pix2pix/pix2pix.py`