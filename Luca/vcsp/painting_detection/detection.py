import numpy as np
import cv2
import copy
from Luca.vcsp.painting_detection.utils import is_painting, get_roi
from Luca.vcsp.utils import multiple_show
from Luca.vcsp.painting_detection.constants import MIN_HULL_AREA_PERCENT
from Luca.vcsp.painting_detection.utils import frame_process
# from Cristian.image_processing.cri_processing_strat import frame_process
from Luca.vcsp.utils.drawing import draw_bb


def get_bb(img, include_steps=False):

    blur, th, morph = frame_process(img)

    _, contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = copy.deepcopy(img)
    rois = []

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

            if is_painting(hull, poly, bounding_box, rotated_box, ellipse, img):
                candidate_hulls.append(hull)
                candidate_polys.append(poly)
                candidate_bounding_boxes.append(bounding_box)

        if len(candidate_hulls) != 0:
            max_hull_area = cv2.contourArea(max(candidate_hulls, key=cv2.contourArea))

            # x*1920*1080+y=0.01
            # x*1280*720+y=0.1
            x = -1 / 12800000
            y = 43 / 250
            img_area = img.shape[0] * img.shape[1]
            img_percent = x * img_area + y
            for i, c in enumerate(candidate_hulls):
                hull_area = cv2.contourArea(candidate_hulls[i])
                if hull_area < max_hull_area * MIN_HULL_AREA_PERCENT or hull_area < img_area * img_percent:
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
                # cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(output, "{}".format(i), (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, False)

                roi = get_roi(candidate_bounding_boxes[index], img)
                rois.append(roi)

    if include_steps:
        hstack1 = multiple_show.horizontal_stack(blur, th)
        hstack2 = multiple_show.horizontal_stack(morph, output)
        output = multiple_show.vertical_stack(hstack1, hstack2)

    return output, rois
