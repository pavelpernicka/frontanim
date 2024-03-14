#!/usr/bin/env python3
import argparse

import PIL.Image

from imgformat import *

parser = argparse.ArgumentParser(description="Show custom metadata")
parser.add_argument("input", help="Input file path")

args = parser.parse_args()

img = PIL.Image.open(args.input)
meta = get_metadata(img)
print(meta)
