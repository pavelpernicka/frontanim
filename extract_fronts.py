#!/usr/bin/env python3
import argparse
import re
from datetime import datetime

import cv2 as cv
import gif2numpy
import numpy as np
import pytesseract
from PIL import Image

from geo import *
from imgformat import *

parser = argparse.ArgumentParser(description="CHMI synoptic situation front extractor")

# positional arguments
parser.add_argument("input", help="Input file path")
parser.add_argument("output", help="Output file path")

# switches
parser.add_argument("--gui", action="store_true", help="Enable GUI mode")
parser.add_argument(
    "--overlay",
    action="store_true",
    help="Only last front is visible in overlaping region",
)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--note", default="", help="Additional note to be saved as EXIF")

args = parser.parse_args()

gui = args.gui
debug = args.debug
overlay_fronts = args.overlay
fn = args.input
output_file = args.output

nadpis_width = 320

winname = "something"

if gui:
    cv.namedWindow(winname)  # Create a named window
    cv.moveWindow(winname, 360, 90)  # Move it to (40,30)


def show(what, force=0):
    if (debug or force) and gui:
        cv.imshow(winname, what)
        cv.waitKey(0)


np_images, extensions, image_specs = gif2numpy.convert(fn)
img = np_images[0]
x, y, _ = img.shape

print(f"Size x: {x}")
print(f"Size y: {y}")

# cover chmi logo
img[x - 39 : x, 0:54] = 0
img[x - 15 : x, 0:y] = 0

# get and parse date using OCR
space = x - nadpis_width
text = img[2:18, space : y - space]
show(text, 1)
config = "-l eng --oem 1 --psm 7"
nadpis = pytesseract.image_to_string(text, config=config)
datum = nadpis.rstrip().split(": ")[1].split(" UTC")[0]
digits = re.findall(r"\d", datum)
datum = "".join(digits)
date = datetime.strptime(datum, "%d%m%Y%H")


print(f"Date: {date.date()}")

# our gif is defaultly in BGR color order
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
show(img_rgb)

# get mast of all fronts
extracted_original_fronts = extract_colors(img_rgb)
all_fronts_mask = (
    extracted_original_fronts[0]
    + extracted_original_fronts[1]
    + extracted_original_fronts[2]
)
show(all_fronts_mask)

# get lines of all fronts together
fronts_lines = extract_front_line(all_fronts_mask)
show(fronts_lines)

# make colorful background image of dilated original fronts
bg_image = np.zeros_like(img_rgb)
for i, front in enumerate(extracted_original_fronts):
    # overwrite pixels where bg_image has zeros
    new_layer = cv.multiply(
        cv.cvtColor(cv.dilate(front, kernel, iterations=3), cv.COLOR_GRAY2RGB),
        colors[i],
    )
    if overlay_fronts:
        black_coords = np.argwhere(np.all(bg_image <= 10, axis=-1))
        black_coords = black_coords[np.all(black_coords < bg_image.shape[:2], axis=1)]
        bg_image[black_coords[:, 0], black_coords[:, 1]] = new_layer[
            black_coords[:, 0], black_coords[:, 1]
        ]
    else:
        bg_image += new_layer
show(bg_image)

# colorise line
show(cv.cvtColor(fronts_lines, cv.COLOR_GRAY2RGB))
colorised_line = cv.bitwise_and(bg_image, bg_image, mask=fronts_lines)

# add ignore frame
black = np.asarray([0, 0, 0])
colorised_line[0:1, 0:y] = black
colorised_line[x - 1 : x, 0:y] = black
colorised_line[0:x, 0:1] = black
colorised_line[0:x, y - 1 : y] = black

show(colorised_line, 1)

image_from_bytes = Image.fromarray(colorised_line)

metadata = {
    "datetime": date,
    "type": "chmi_analysis",
    "projection": "original",
    "note": args.note,
}
exif = exif_from_metadata(metadata)

image_from_bytes.save(output_file, format="PNG", exif=exif)
