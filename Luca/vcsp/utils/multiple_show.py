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


def show_on_row(img1, img2):
    height = max([img1.shape[0], img2.shape[0]])
    width = img1.shape[1] + img2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)

    output[0:img1.shape[0], 0:img1.shape[1], :] = img1
    output[0:img2.shape[0], img1.shape[1]:, :] = img2

    output = resize_to_fit(output)

    return output


def show_on_col(img1, img2):
    height = img1.shape[0] + img2.shape[0]
    width = max([img1.shape[1], img2.shape[1]])

    output = np.zeros((height, width, 3), dtype=np.uint8)

    output[0:img1.shape[0], 0:img1.shape[1], :] = img1
    output[img1.shape[0]:, 0:img2.shape[1]:, :] = img2

    output = resize_to_fit(output)

    return output


def resize_to_fit(img):
    h, w = img.shape[0:2]
    dist_w = max(0, w - 1920)
    dist_h = max(0, h - 1080)

    if dist_h == 0 and dist_w == 0:
        return img

    if dist_h > dist_w:
        scale_factor = (1 - dist_h / h)
    else:
        scale_factor = (1 - dist_w / w)

    width = int(w * scale_factor)
    height = int(h * scale_factor)
    output = cv2.resize(img, (width, height))
    return output


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