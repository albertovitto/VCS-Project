import numpy as np
import cv2
import copy

from scipy.spatial import distance

from Luca.vcsp.painting_rectification.utils import get_four_coordinates
from Luca.vcsp.utils import multiple_show
from Alberto.could_not_find_matches import could_not_find_matches
from Luca.vcsp.utils.multiple_show import resize_to_fit


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


def rectify_with_retrieval(img, ground_truth):
    sift_img = cv2.xfeatures2d_SIFT.create()
    kp_img, des_img = sift_img.detectAndCompute(img, None)
    out_img = cv2.drawKeypoints(img, kp_img, None)

    sift_ground_truth = cv2.xfeatures2d_SIFT.create()
    kp_ground_truth, des_ground_truth = sift_ground_truth.detectAndCompute(ground_truth, None)
    out_ground_truth = cv2.drawKeypoints(ground_truth, kp_ground_truth, None)

    MIN_MATCH_COUNT = 4
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_img, des_ground_truth, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ground_truth[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        h, w, d = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        ground_truth = cv2.polylines(ground_truth, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matches_mask = None

    rectified = None

    if matches_mask:
        ground_truth_h, ground_truth_w, _ = ground_truth.shape
        rectified = cv2.warpPerspective(src=img, M=M, dsize=(ground_truth_w, ground_truth_h))
        rectified = resize_to_fit(rectified, dh=img.shape[0], dw=img.shape[1])

    output = cv2.drawMatches(img, kp_img, ground_truth, kp_ground_truth, good, None,
                          matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)

    if matches_mask is None:
        h, w, c = img.shape
        output = np.hstack((img,could_not_find_matches(h,w,c)))
    return rectified, output