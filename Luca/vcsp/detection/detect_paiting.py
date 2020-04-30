import numpy as np
import cv2
from ..utils.brightness_contrast import auto_alpha_beta
from ..utils.multiple_show import horizontal_stack, vertical_stack
from .utils import is_painting, remove_duplicated
from skimage import filters


def detect_painting(img, include_steps=False):
    alpha, beta = auto_alpha_beta(img)
    high_contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    gray_img = cv2.cvtColor(high_contrast_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 5)

    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((5, 5)), iterations=1)

    _, contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        maxc = cv2.contourArea(max(contours, key=cv2.contourArea))
        fixed_contours = []
        rects = []

        for i, c in enumerate(contours):
            hull = cv2.convexHull(c)
            epsilon = (len(c) / (3 * 4)) * 2  # 4 == number of desired points
            contour_poly = cv2.approxPolyDP(hull, epsilon, True)

            if is_painting(contour_poly, img, maxc):
                fixed_contours.append(contour_poly)

        for i, c in enumerate(fixed_contours):
            r = cv2.boundingRect(c)
            rects.append(r)
            x, y, w, h = r
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rects = remove_duplicated(rects)

        for i, r in enumerate(rects):
            x, y, w, h = r

            cv2.drawContours(img, [c], 0, (0, 0, 255), 2)
            # cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)
            #cv2.drawContours(img, [contour_poly], 0, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


    if include_steps:
        hstack1 = horizontal_stack(blur, th)
        hstack2 = horizontal_stack(morph, img)
        vstack = vertical_stack(hstack1, hstack2)
        return vstack
    return img
