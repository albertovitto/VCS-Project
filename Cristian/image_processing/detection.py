import cv2
import numpy as np
from Luca.vcsp.utils.multiple_show import horizontal_stack, vertical_stack
from Luca.vcsp.detection.utils import is_painting, remove_duplicated


def get_bb(img):
    ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(ret, 5, 75, 75)
    th = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 5)
    # _, th = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th = cv2.bitwise_not(th)
    # canny = cv2.Canny(th, 50, 100, 5)
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)
    morph = cv2.dilate(morph, np.ones((3, 3)))

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
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rects = remove_duplicated(rects)

        for i, r in enumerate(rects):
            x, y, w, h = r

            cv2.drawContours(img, [c], 0, (0, 0, 255), 2)
            cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)
            cv2.drawContours(img, [contour_poly], 0, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    hstack1 = horizontal_stack(filtered, th)
    hstack2 = horizontal_stack(morph, img)
    vstack = vertical_stack(hstack1, hstack2)
    return vstack
