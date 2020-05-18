import numpy as np
import cv2
from Luca.vcsp.painting_detection.constants import MIN_HULL_POINTS, MIN_POLY_POINTS, MIN_ROTATED_BOX_AREA_PERCENT, \
    MIN_ROTATED_ELLIPSE_AREA_PERCENT, MIN_POLY_AREA_PERCENT, MAX_RATIO_SIZE, MAX_GRAY_60_PERCENTILE, MIN_VARIANCE


def frame_process(img):
    alpha, beta = auto_alpha_beta(img)
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 5)

    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((5, 5)), iterations=1)

    return blur, th, morph


def get_roi(bounding_box, img):
    x, y, w, h = bounding_box
    mask = np.zeros_like(img[:, :, 0])
    mask[y:y + h, x:x + w] = 1
    roi = cv2.bitwise_and(img, img, mask=mask)
    roi = roi[y:y + h, x:x + w]
    return roi


def is_painting(hull, poly, bounding_box, rotated_box, ellipse, img):
    # se CONVEX HULL e APPROXPOLY hanno meno di 3 vertici
    if len(hull) <= MIN_HULL_POINTS or len(poly) <= MIN_POLY_POINTS:
        return False

    # se l'area di HULL è più piccola dell'80% (75%) dell'area del ROTATED MIN RECT
    if cv2.contourArea(hull) < cv2.contourArea(rotated_box) * MIN_ROTATED_BOX_AREA_PERCENT:  # non è un rettangolo
        if ellipse is None:
            return False
        else:
            (x, y), (MA, ma), angle = ellipse
            if cv2.contourArea(hull) < np.pi * MA/2 * ma/2 * MIN_ROTATED_ELLIPSE_AREA_PERCENT:  # ma neanche un cerchio
                return False

    # se l'area di POLY è più piccola dell'70% (60%) dell'area di HULL
    if cv2.contourArea(poly) < cv2.contourArea(hull) * MIN_POLY_AREA_PERCENT:
        return False

    # se il rapporto WIDTH / HEIGHT del BB è superiore a 3
    x, y, w, h = bounding_box
    if w >= MAX_RATIO_SIZE * h or h >= MAX_RATIO_SIZE * w:
        return False

    # se il ROI è troppo luminoso/bianco (60% maggiore di 175)
    roi = get_roi(bounding_box, img)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if np.percentile(gray_roi, 60) >= MAX_GRAY_60_PERCENTILE:
        return False

    # se il ROI ha una varianza troppo bassa (minore di 10)
    if np.std(gray_roi) < MIN_VARIANCE:
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


def simplify_contour(contour, n_corners=4):
    """
    Binary searches best `epsilon` value to force contour
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    """
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx
