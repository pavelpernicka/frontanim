#!/bin/bash

data_files="datasets/generated_model-v4/data"
fronts_files="datasets/generated_model-v4/fronts"
files=$data_files/*

suicide() {
  echo "Exiting due to Ctrl+C"
  pkill -P $$
  exit 1
}

trap suicide INT

for data_image in $files; do
    echo $data_image
    fname=$(basename "${data_image%.*}")
    front_image="$fronts_files/$fname.png"
    
    for rotation_angle in "90" "180" "270"; do
        #rotation_angle=$(shuf -i 0-360 -n 1)
        data_image_out="$data_files/$fname-$rotation_angle.png"
        fronts_image_out="$fronts_files/$fname-$rotation_angle.png" 
        echo "Rotating $fname -> $rotation_angle degrees"
        convert "$data_image" -background white -rotate "$rotation_angle" "$data_image_out"
        convert "$front_image" -background white -rotate "$rotation_angle" "$fronts_image_out"
    done
done


trap - INT