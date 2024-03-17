#!/usr/bin/env bash

geojson_files="datasets/generated_overlay/geojson/*"
target_folder="datasets/generated_model"

mkdir -p "$target_folder"
mkdir -p "$target_folder/data"
mkdir -p "$target_folder/fronts"

skip_existing=false  # Set to true if you want to skip existing files

suicide() {
  echo "Exiting due to Ctrl+C"
  pkill -P $$
  exit 1
}

trap suicide INT

for filename in $geojson_files; do
    fname=$(basename "${filename%.*}")
    
    data_file="$target_folder/data/$fname.png"
    fronts_file="$target_folder/fronts/$fname.png"
    
    echo "Processing $filename..."
    
    if [[ ! -f "$data_file" || ( -f "$data_file" && ! $skip_existing ) ]]; then
        ./grib_reader.py --source "$filename" --save "$data_file"
    fi
    
    if [[ ! -f "$fronts_file" || ( -f "$fronts_file" && ! $skip_existing ) ]]; then
        ./grib_reader.py --source "$filename" --fronts --nodata --save "$fronts_file"
    fi
done

trap - INT