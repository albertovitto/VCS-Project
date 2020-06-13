import cv2
import numpy as np


def frame_preprocess(img, params):
    alpha, beta = auto_alpha_beta(img)
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray_img, 5, 75, 75)

    block_size = int(np.ceil(img.shape[1] / params["THRESHOLD_BLOCK_SIZE_FACTOR"]) // 2 * 2 + 1)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, params["THRESHOLD_C"])

    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=4)
    # morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)

    return blur, th, morph


def get_roi(bounding_box, img):
    x, y, w, h = bounding_box
    mask = np.zeros_like(img[:, :, 0])
    mask[y:y + h, x:x + w] = 1
    roi = cv2.bitwise_and(img, img, mask=mask)
    roi = roi[y:y + h, x:x + w]
    return roi


def is_painting(hull, poly, bounding_box, rotated_box, ellipse, img, params):
    if len(hull) <= params["MIN_HULL_POINTS"] or len(poly) <= params["MIN_POLY_POINTS"]:
        return False

    if cv2.contourArea(hull) < cv2.contourArea(rotated_box) * params["MIN_ROTATED_BOX_AREA_PERCENT"]:
        if ellipse is None:
            return False
        else:
            (x, y), (MA, ma), angle = ellipse
            if cv2.contourArea(hull) < np.pi * MA/2 * ma/2 * params["MIN_ROTATED_ELLIPSE_AREA_PERCENT"]:
                return False

    if cv2.contourArea(poly) < cv2.contourArea(hull) * params["MIN_POLY_AREA_PERCENT"]:
        return False

    x, y, w, h = bounding_box
    if w >= params["MAX_RATIO_SIZE"] * h or h >= params["MAX_RATIO_SIZE"] * w:
        return False

    roi = get_roi(bounding_box, img)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if np.percentile(gray_roi, 80) >= params["MAX_GRAY_80_PERCENTILE"]:
        return False

    blur_roi = cv2.GaussianBlur(roi, (7, 7), 0)
    if np.mean(cv2.meanStdDev(blur_roi)[1]) < params["MIN_VARIANCE"]:
        return False

    return True


def auto_alpha_beta(img):
    # 1 method
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    brightness = hsv_planes[2]
    mean, std = cv2.meanStdDev(brightness)

    beta = std - 10
    if beta < 0:
        beta = 0

    alpha = 1.2

    # 2 method
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    percent = np.percentile(gray_img, 90)
    alpha = (255 / percent)
    beta = 0
    #beta = beta / 2
    alpha = 2

    return alpha, beta