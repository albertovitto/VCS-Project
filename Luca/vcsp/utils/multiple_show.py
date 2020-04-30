import numpy as np
import cv2


def horizontal_stack(img1, img2):
    width = np.rint(img1.shape[0] / 2).astype(np.uint16)
    height = img1.shape[1]

    if len(img1.shape) == 2:
        i1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), (height, width))
    else:
        i1 = cv2.resize(img1, (height, width))

    if len(img2.shape) == 2:
        i2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), (height, width))
    else:
        i2 = cv2.resize(img2, (height, width))

    hstack = np.hstack((i1, i2))
    return hstack


def vertical_stack(img1, img2):
    width = img1.shape[0]
    height = np.rint(img1.shape[1] / 2).astype(np.uint16)

    if len(img1.shape) == 2:
        i1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), (height, width))
    else:
        i1 = cv2.resize(img1, (height, width))

    if len(img2.shape) == 2:
        i2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), (height, width))
    else:
        i2 = cv2.resize(img2, (height, width))

    vstack = np.vstack((i1, i2))
    return vstack




"""
width = np.rint(img.shape[0] / 2).astype(np.uint16)
height = np.rint(img.shape[1] / 2).astype(np.uint16)

i1 = cv2.resize(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), (height, width))
i2 = cv2.resize(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR), (height, width))
i3 = cv2.resize(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR), (height, width))
i4 = cv2.resize(img, (height, width))

horizontal_show1 = np.hstack((i1, i2))
horizontal_show2 = np.hstack((i3, i4))
vertical_show = np.vstack((horizontal_show1, horizontal_show2))
cv2.imshow('Frame', vertical_show)
"""