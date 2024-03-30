import cv2 as cv
import numpy as np

from colors import *

kernel = np.ones((4, 4), np.uint8)  # 4x4 fits just the best
kernel1 = np.ones((3, 3), np.uint8)  # 4x4 fits just the best
kernel2 = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]], np.uint8)


def colorise_fronts(image):
    output = []
    for i, img in enumerate(image):
        output.append(cv.multiply(cv.cvtColor(img, cv.COLOR_GRAY2RGB), colors[i]))
    return output


def extract_colors(image, color_definition=chmi_colors, tolerance=50):
    fronts = []
    # extract fronts by color, create mask
    for colord in color_definition:
        fronts.append(
            cv.inRange(
                image, colord - tolerance, colord + tolerance
            )
        )
    return fronts


def extract_front_line(mask_in, nr_iterations=10):
    skeleton_new = np.zeros_like(mask_in)
    num_labels, labels = cv.connectedComponents(mask_in)
    # must skeletonize each chunk of fronts individually, otherwise e.g. secondary front will be connected with main front
    for label_id in range(num_labels):
        mask = (labels == label_id + 1).astype("uint8") * 255

        # get spikes
        # https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
        mask = cv.dilate(mask, kernel, iterations=2)
        spikes = cv.erode(mask, kernel1, iterations=1)

        # skeletonize
        # https://docs.opencv.org/3.4/df/d2d/group__ximgproc.html
        # thinning types (arg 1):
        # 0=THINNING_ZHANGSUEN
        # 1=THINNING_GUOHALL
        skeleton = cv.ximgproc.thinning(mask, 1)
        skeleton_erodet = np.copy(skeleton)

        for i in range(nr_iterations):
            skeleton_erodet = cv.dilate(skeleton_erodet, kernel1, iterations=1)
            skeleton_erodet = cv.erode(skeleton_erodet, kernel2, iterations=1)

        skeleton_new += cv.ximgproc.thinning(skeleton_erodet, 1)
    return skeleton_new  # return result
