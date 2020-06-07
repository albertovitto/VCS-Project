import copy
import cv2
import numpy as np

from estensi.painting_detection.constants import conf
from estensi.painting_detection.utils import is_painting, frame_preprocess, get_roi
from estensi.utils import draw_bb, show_on_col, show_on_row


def get_bb(img, params=conf, include_steps=False):
    blur, th, morph = frame_preprocess(img, params)

    _, contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = copy.deepcopy(img)
    rois = []
    bbs = []

    if len(contours) != 0:
        candidate_bounding_boxes = []
        candidate_hulls = []
        candidate_polys = []
        found = []

        for i, c in enumerate(contours):
            hull = cv2.convexHull(c)
            epsilon = (len(c) / (3 * 4)) * 2  # 4 == number of desired points
            poly = cv2.approxPolyDP(c, epsilon, True)
            rotated_box = cv2.minAreaRect(hull)
            rotated_box = cv2.boxPoints(rotated_box)
            rotated_box = np.int0(rotated_box)
            bounding_box = cv2.boundingRect(hull)
            ellipse = None
            if len(hull) > 5:
                ellipse = cv2.fitEllipse(hull)

            if is_painting(hull, poly, bounding_box, rotated_box, ellipse, img, params):
                candidate_hulls.append(hull)
                candidate_polys.append(poly)
                candidate_bounding_boxes.append(bounding_box)

        if len(candidate_hulls) != 0:
            max_hull_area = cv2.contourArea(max(candidate_hulls, key=cv2.contourArea))

            img_area = img.shape[0] * img.shape[1]
            for i, c in enumerate(candidate_hulls):
                hull_area = cv2.contourArea(candidate_hulls[i])
                if hull_area < max_hull_area * params["MIN_HULL_AREA_PERCENT_OF_MAX_HULL"] \
                        or hull_area < img_area * params["MIN_HULL_AREA_PERCENT_OF_IMG"]:
                    continue
                else:
                    found.append(i)

        if len(found) != 0:
            for i, index in enumerate(found):
                x, y, w, h = candidate_bounding_boxes[index]

                if include_steps:
                    cv2.drawContours(output, [candidate_hulls[index]], 0, (0, 0, 255), 2)
                    cv2.drawContours(output, [candidate_polys[index]], 0, (0, 255, 0), 2)

                draw_bb(output, tl=(x, y), br=(x + w, y + h), color=(255, 0, 0), label="painting")

                roi = get_roi(candidate_bounding_boxes[index], img)
                rois.append(roi)
                bbs.append(list(candidate_bounding_boxes[index]))  # tuple non è modificabile => list

    if include_steps:
        hstack1 = show_on_row(blur, th)
        hstack2 = show_on_row(morph, output)
        output = show_on_col(hstack1, hstack2)

    return output, rois, bbs
