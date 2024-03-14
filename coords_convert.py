#!/usr/bin/env python3
from osgeo import osr


class CoordsConverter:
    "Converter from ESRI:102031 to WGS84"

    def __init__(self):
        proj4_string = "+proj=eqdc +lat_0=30 +lon_0=10 +lat_1=43 +lat_2=62 +x_0=0 +y_0=0 +ellps=intl +units=m +no_defs"  # ESRI:102031 (not present in osgeo, copied from QGIS)

        src_srs = osr.SpatialReference()
        src_srs.ImportFromProj4(proj4_string)

        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)  # EPSG code for WGS84

        self.transform = osr.CoordinateTransformation(src_srs, dst_srs)

    def coords(self, coord):
        "Convert coordinates"
        transformed_coords = self.transform.TransformPoint(coord[0], coord[1])

        lat = transformed_coords[0]
        lon = transformed_coords[1]

        return (lat, lon)

    def std_string(self, coord):
        if coord[0] >= 0:
            lat_sign = "N"
        else:
            lat_sign = "S"

        if coord[1] >= 0:
            lon_sign = "E"
        else:
            lon_sign = "W"

        return f"{coord[0]}{lat_sign}, {coord[1]}{lon_sign}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert EPSG:102031 (meters) coordinates to WGS84 (degrees)"
    )

    # positional arguments
    parser.add_argument("x", help="X coordinate", type=float)
    parser.add_argument("y", help="Y coordinate", type=float)
    # switches
    parser.add_argument("--raw", action="store_true", help="No N,S,W,E suffixes")
    args = parser.parse_args()

    coord = (args.x, args.y)
    conv = CoordsConverter()
    try:
        converted = conv.coords(coord)
    except TypeError:
        print("Invalid input coordinates")
    else:
        lat, lon = converted
        if args.raw:
            print(f"{lat}, {lon}")
        else:
            print(conv.std_string(converted))
