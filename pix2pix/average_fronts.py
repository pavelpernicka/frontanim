#!/usr/bin/env python3
import os

import cv2
import numpy as np


def calculate_average_image(folder_path):
    # Get a list of all PNG files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

    # Check if there are any images in the folder
    if not image_files:
        print("No PNG images found in the folder.")
        return

    # Load the first image to initialize the sum
    first_image_path = os.path.join(folder_path, image_files[0])
    sum_image = cv2.imread(first_image_path, cv2.IMREAD_COLOR).astype(np.float64)

    # Loop through the remaining images and add them to the sum
    ilen = 0
    for image_file in image_files[1:]:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float64)
        if image.shape[0] == 561 and image.shape[1] == 760:
            sum_image += image
            print("ok")
            ilen += 1
        # cv2.imshow("Image", sum_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Calculate the average
    average_image = sum_image / ilen

    return average_image


if __name__ == "__main__":
    folder_path = "png"

    average_image = calculate_average_image(folder_path)

    if average_image is not None:
        cv2.imshow("Average Image", average_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
