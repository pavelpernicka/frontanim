#!/bin/bash

path=datasets/synoptic_analyse_original/*.gif
overlay=""
skip_existing=false
target_folder=""
note="test_convert"

help () {
  echo "usage: $0 [--overlay | -o] [--skip_existing | -s] [--note | -n NOTE] [--source_path | -p PATH] <target_folder>"
  echo ""
  echo "Convert all synoptic situations from source dataset"
  exit 1
}

suicide() {
  echo "Exiting due to Ctrl+C"
  pkill -P $$
  exit 1
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -o|--overlay)
            overlay="--overlay"
            shift
            ;;
        -s|--skip_existing)
            skip_existing=true
            shift
            ;;
        -h|--help)
            help
            shift
            ;;
        -n|--note)
            note="$2"
            shift
            shift
            ;;
        -p|--source_path)
            path="$2"
            shift
            shift
            ;;
        *)
            target_folder="$1"
            shift
            ;;
    esac
done

if [[ -z $target_folder ]]; then
    echo "Error: Please specify a target folder."
    help
fi

echo "Processing files in: $path"
echo "Skip existing: $skip_existing"
echo "Target folder: $target_folder"
echo "Note: $note"

for folder_name in "png" "tif" "geojson"; do
    folder=$target_folder/$folder_name
    if [[ ! -d "$folder" ]]; then
        if ! mkdir -p "$folder"; then
            echo "Failed to create folder: $folder"
            exit 1
        fi
    fi
done


trap suicide INT
for filename in $path; do
    fname=$(basename "${filename%.*}")
    
    png_file="$target_folder/png/$fname.png"
    geojson_file="$target_folder/geojson/$fname.geojson"
    tif_file="$target_folder/tif/$fname.tif"
    
    echo "Processing $filename..."
    
    if [[ ! -f "$png_file" || ( -f "$png_file" && ! $skip_existing ) ]]; then
        ./extract_fronts.py --note "$note" $overlay "$filename" "$png_file"
    fi
    
    if [[ ! -f "$geojson_file" || ( -f "$geojson_file" && ! $skip_existing ) ]]; then
        ./vectorize.py --export_tif "$tif_file" "$png_file" "$geojson_file" 
    fi
done
trap - INT