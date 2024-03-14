#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile

import cv2 as cv
import PIL.Image
from geojson import Feature, FeatureCollection, LineString, dump
from osgeo import gdal

from coords_convert import CoordsConverter
from geo import *
from imgformat import *

precision = 0.008

parser = argparse.ArgumentParser(description="Raster front linies to geojson converter")

# positional arguments
parser.add_argument("input", help="Input file path")
parser.add_argument("output", help="Output file path")

# switches
parser.add_argument("--gui", action="store_true", help="Enable GUI mode")
parser.add_argument("--debug", action="store_true", help="Show detailed info")
parser.add_argument("--export_tif", help="Save geotiff as [file name]")

args = parser.parse_args()
coord_converter = CoordsConverter()
script_path = os.path.dirname(os.path.realpath(__file__))

winname = "sth"

if args.gui:
    cv.namedWindow(winname)  # Create a named window
    cv.moveWindow(winname, 360, 90)  # Move it to (40,30)


def show(what, force=0):
    if args.gui:
        cv.imshow(winname, what)
        cv.waitKey(0)


georeferenced = tempfile.NamedTemporaryFile()
print("Georeferencing image...")
os.system(f"{script_path}/georef.sh {args.input} {georeferenced.name}")
img_for_meta = PIL.Image.open(args.input)
meta = get_metadata(img_for_meta)
img_for_meta.close()

gdal_ds = gdal.Open(georeferenced.name)
if args.export_tif:
    shutil.copyfile(georeferenced.name, args.export_tif)

xoff, a, b, yoff, d, e = gdal_ds.GetGeoTransform()


def lat_lon(point):
    x, y = point
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    converted = coord_converter.coords((xp, yp))
    lat, lon = converted
    return (lon, lat)


print(f"Image metadata: {meta}")

colorised_line = cv.imread(georeferenced.name)
show(colorised_line)

# extract fronts by color from final line
fronts_by_types = extract_colors(colorised_line, colors, 254)
for i, front in enumerate(fronts_by_types):
    fronts_by_types[i] = extract_front_line(front)
    show(fronts_by_types[i])

geojson_features = []


def get_nearest_point(current_point, points):
    distances = np.linalg.norm(points - current_point, axis=1)
    min_distance_index = np.argmin(distances)
    return min_distance_index


def arrange_points(points):
    ordered_points = []
    remaining_points = np.copy(points)

    start_point_index = np.argmin(points[:, 0])  # FIXME: isnt allways that
    current_point = points[start_point_index]
    ordered_points.append(current_point)
    remaining_points = np.delete(remaining_points, start_point_index, axis=0)

    while len(remaining_points) > 0:
        nearest_index = get_nearest_point(current_point, remaining_points)
        nearest_point = remaining_points[nearest_index]
        ordered_points.append(nearest_point)
        current_point = nearest_point
        remaining_points = np.delete(remaining_points, nearest_index, axis=0)

    return np.array(ordered_points)


for front_id, front_type in enumerate(fronts_by_types):
    show(front_type)
    ret, thresh = cv.threshold(front_type, 127, 255, 0)
    # contours, hierarchy = cv.findContours(front_dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    num_labels, labels = cv.connectedComponents(thresh)
    print(f"Found {num_labels} individual front lines in frontset {front_id}")
    contour_id = 0
    for label_id in range(num_labels):
        print(f"Processing cc: {label_id}")
        connected_component = (labels == label_id + 1).astype("uint8") * 255
        show(connected_component)
        contour = np.column_stack(np.where(connected_component.transpose() != 0))
        print(contour)
        if len(contour) >= 3:
            contour = arrange_points(contour)
            print(contour)
            epsilon = precision * cv.arcLength(contour, True)
            polyline_coords = cv.approxPolyDP(contour, epsilon, closed=False)
            converted_coordinates = [lat_lon(coord[0]) for coord in polyline_coords]
            if args.debug:
                print(converted_coordinates)
            geometry = LineString(converted_coordinates)
            front_meta = {"front_type": front_names[front_id], "front_id": contour_id}
            feature = Feature(geometry=geometry, properties=front_meta)
            geojson_features.append(feature)

            # draw it (if enabled)
            if args.gui:
                polyline_image = np.zeros_like(thresh)
                cv.polylines(
                    polyline_image,
                    [polyline_coords],
                    isClosed=False,
                    color=255,
                    thickness=1,
                )
                for point in polyline_coords:
                    polyline_image = cv.circle(polyline_image, point[0], 4, 255, -1)
                    show(polyline_image)
                show(polyline_image, 1)
            contour_id += 1
        else:
            print("Front area too small")

# save output
geojson = FeatureCollection(features=geojson_features, properties=meta)
print("\nOutput geojson:")
print(geojson)

try:
    f = open(args.output, "w")
    dump(geojson, f, indent=4)
    print(f"GeoJSON file saved successfully: {args.output}")
except (IOError, OSError) as e:
    print(f"Error creating GeoJSON file: {e}")
