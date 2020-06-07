import numpy as np


def get_four_coordinates(contour):
    # list to hold ROI coordinates
    rect = np.zeros((4, 2), dtype="float32")
    # top left corner will have the smallest sum,
    # bottom right corner will have the largest sum
    s = np.sum(contour, axis=1)
    rect[0] = contour[np.argmin(s)]
    rect[2] = contour[np.argmax(s)]
    # top-right will have smallest difference
    # botton left will have largest difference
    diff = np.diff(contour, axis=1)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]

    return rect
