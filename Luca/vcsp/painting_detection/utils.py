import numpy as np
import cv2


def get_roi(bounding_box, img):
    x, y, w, h = bounding_box
    mask = np.zeros_like(img[:, :, 0])
    mask[y:y + h, x:x + w] = 1
    roi = cv2.bitwise_and(img, img, mask=mask)
    roi = roi[y:y + h, x:x + w]
    return roi


def is_painting(hull, poly, bounding_box, rotated_box, img):
    # se CONVEX HULL e APPROXPOLY hanno meno di 3 vertici
    if len(hull) <= 3 or len(poly) <= 3:
        return False

    # se l'area di HULL è più piccola dell'80% (75%) dell'area del ROTATED MIN RECT
    if cv2.contourArea(hull) < cv2.contourArea(rotated_box) * 0.75:
        return False

    # se l'area di POLY è più piccola dell'70% (60%) dell'area di HULL
    if cv2.contourArea(poly) < cv2.contourArea(hull) * 0.6:
        return False

    # se il rapporto WIDTH / HEIGHT del BB è superiore a 3
    x, y, w, h = bounding_box
    if w >= 3 * h or h >= 3 * w:
        return False

    # se il ROI è troppo luminoso/bianco (60% maggiore di 175)
    roi = get_roi(bounding_box, img)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if np.percentile(gray_roi, 60) >= 175:
        return False

    # se il ROI ha una varianza troppo bassa (minore di 10)
    if np.std(gray_roi) < 10:
        return False

    # se 3 lati del BB sono i bordi dell'immagine
    if x == 0 and h == img.shape[0]:
        return False
    if x + w == img.shape[1] and h == img.shape[0]:
        return False
    if y == 0 and w == img.shape[1]:
        return False
    if y + h == img.shape[0] and w == img.shape[1]:
        return False

    # altrimenti
    return True


def auto_alpha_beta(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    brightness = hsv_planes[2]
    mean, std = cv2.meanStdDev(brightness)

    beta = std - 10
    if beta < 0:
        beta = 0

    alpha = 1.2

    return alpha, beta
