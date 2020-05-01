import cv2
import numpy as np
from .multiple_show import horizontal_stack, vertical_stack


def is_painting(contour, img, max_area):
    contour_area = cv2.contourArea(contour)

    if contour_area < max_area * 0.05:
        return False

    if not cv2.isContourConvex(contour):
        return False

    """if len(contour) < 4:
        return False"""

    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros_like(img[:, :, 0])
    mask[y:y + h, x:x + w] = 1
    roi = cv2.bitwise_and(img, img, mask=mask)
    roi = roi[y:y + h, x:x + w]
    if is_background(roi):
        return False

    return True


def is_background(roi):
    means, stds = cv2.meanStdDev(roi)

    mean = np.mean(means)
    std = np.mean(stds)

    if std < 30:
        return True

    return False


def remove_duplicated(bounding_boxes):
    to_remove = []

    for i, c in enumerate(bounding_boxes):
        for j, other in enumerate(bounding_boxes):
            # se è lo stesso box, continua
            if i == j:
                continue
            # altrimenti
            # calcola l'intersezione
            intersection = check_intersection(c, other)
            # se l'intersezione non è vuota
            if intersection != (0, 0, 0, 0):
                # se l'intersezione copre una certa percentuale di area
                if intersection[2] * intersection[3] > c[2] * c[3] * 0.8:
                    # togliamo il box più piccolo
                    if c[2] * c[3] > other[2] * other[3]:
                        to_remove.append(j)
                    else:
                        to_remove.append(i)

    for j in to_remove:
        bounding_boxes.pop(j)

    return bounding_boxes


def check_intersection(box1, box2):
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("box must be a tuple (x, y, w, h)")

    x = max(box1[0], box2[0])
    y = max(box1[1], box2[1])
    w = min(box1[0] + box1[2], box2[0] + box2[2]) - x
    h = min(box1[1] + box1[3], box2[1] + box2[3]) - y

    if w < 0 or h < 0:
        return 0, 0, 0, 0

    return x, y, w, h


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
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # rects = remove_duplicated(rects)

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
