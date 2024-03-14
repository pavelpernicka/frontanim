from PIL.Image import Exif

metadata_structure = ["datetime", "type", "projection", "note"]


def get_metadata(img):
    exif = img._getexif()
    metadata = dict()
    for i in exif:
        metadata[metadata_structure[i]] = exif[i]
    return metadata


def exif_from_metadata(metadata):
    exif = Exif()
    for i, key in enumerate(metadata_structure):
        if key == "datetime":
            exif[i] = int(metadata[key].timestamp())
        else:
            exif[i] = metadata[key]
    return exif
