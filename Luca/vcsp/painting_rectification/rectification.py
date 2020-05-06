import numpy as np
import cv2
import copy

from scipy.spatial import distance

from Luca.vcsp.painting_rectification.utils import get_four_coordinates
from Luca.vcsp.utils import multiple_show


def rectify(img, include_steps=False):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 5)

    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((5, 5)), iterations=1)

    _, contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = copy.deepcopy(img)

    if len(contours) != 0:
        candidate_hulls = []
        candidate_polys = []

        for i, c in enumerate(contours):
            hull = cv2.convexHull(c)
            epsilon = (len(c) / (3 * 4)) * 2  # 4 == number of desired points
            poly = cv2.approxPolyDP(c, epsilon, True)

            candidate_hulls.append(hull.reshape(hull.shape[0], hull.shape[2]))
            candidate_polys.append(poly.reshape(poly.shape[0], poly.shape[2]))

            cv2.drawContours(output, [hull], 0, (0, 0, 255), 2)
            cv2.drawContours(output, [poly], 0, (255, 0, 0), 2)

        hull = sorted(candidate_hulls, key=cv2.contourArea, reverse=True)[0]

        rect = get_four_coordinates(hull)
        (tr, tl, br, bl) = rect

        top_width = distance.euclidean(tl, tr)
        bottom_width = distance.euclidean(bl, br)
        max_width = max(int(top_width), int(bottom_width))

        left_height = distance.euclidean(tl, bl)
        right_height = distance.euclidean(br, tr)
        max_height = max(int(left_height), int(right_height))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        transform_matrix, _ = cv2.findHomography(rect, dst, cv2.RANSAC, 5.0)

        result = cv2.warpPerspective(img, transform_matrix, (max_width, max_height))

        if include_steps:
            hstack1 = multiple_show.horizontal_stack(blur, th)
            hstack2 = multiple_show.horizontal_stack(morph, output)
            output = multiple_show.vertical_stack(hstack1, hstack2)
        else:
            output = result
        return output




