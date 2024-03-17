#!/usr/bin/env bash

geojson_files="datasets/generated_overlay/geojson/*"
target_folder="datasets/generated_model"
exclude_list=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --exclude_list)
            exclude_list="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

exclude=()
if [ -n "$exclude_list" ]; then
    while IFS= read -r line; do
        exclude+=("$line")  # Remove the extra space after $line
    done < "$exclude_list"
fi

for filename in $geojson_files; do
    fname=$(basename "${filename%.*}")
    found=false
    for item in "${exclude[@]}"; do
        if [ "$fname" = "$item" ]; then
            found=true
            break
        fi
    done

    if [ "$found" = true ]; then
        echo "$fname is in exclude list, skipping"
    else
        data_file="$target_folder/data/$fname.png"
        fronts_file="$target_folder/fronts/$fname.png"
        
        echo "Processing $filename..."
        
        if [[ ! -f "$data_file" || ( -f "$data_file" && ! $skip_existing ) ]]; then
            ./grib_reader.py --source "$filename" --save "$data_file"
        fi
        
        if [[ ! -f "$fronts_file" || ( -f "$fronts_file" && ! $skip_existing ) ]]; then
            ./grib_reader.py --source "$filename" --fronts --nodata --save "$fronts_file"
        fi
    fi
done

trap - INT