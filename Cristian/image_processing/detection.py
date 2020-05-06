import cv2
import numpy as np
from Luca.vcsp.utils.multiple_show import horizontal_stack, vertical_stack
from Luca.vcsp.detection.utils import is_painting, remove_duplicated
from Cristian.image_processing.cri_processing_strat import frame_process


def get_bb(img, include_steps=True):
    blur, th, morph = frame_process(img)
    _, contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []

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

        rects = remove_duplicated(rects)
        output = np.ndarray.copy(img)
        for i, r in enumerate(rects):
            x, y, w, h = r
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rois.append(img[y:y + h, x:x + w])

    if include_steps:
        hstack1 = horizontal_stack(blur, th)
        hstack2 = horizontal_stack(morph, output)
        output = vertical_stack(hstack1, hstack2)
    return output, rois
