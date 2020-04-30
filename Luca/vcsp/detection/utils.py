import numpy as np
import cv2


def is_painting(contour, img, max_area):

    contour_area = cv2.contourArea(contour)

    if contour_area < max_area*0.05:
        return False

    if not cv2.isContourConvex(contour):
        return False

    """if len(contour) < 4:
        return False"""

    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros_like(img[:, :, 0])
    mask[y:y + h, x:x + w] = 1
    roi = cv2.bitwise_and(img, img, mask=mask)
    roi = roi[y:y+h, x:x+w]
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
                if intersection[2]*intersection[3] > c[2]*c[3]*0.8:
                    # togliamo il box più piccolo
                    if c[2]*c[3] > other[2]*other[3]:
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



