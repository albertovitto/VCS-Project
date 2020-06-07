import cv2
import numpy as np
import copy
from scipy.spatial import distance
from estensi.painting_rectification.utils import get_four_coordinates
from estensi.utils import show_on_row, show_on_col


def sift_feature_matching_and_homography(roi, img, include_steps=False):
    # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    sift_roi = cv2.xfeatures2d_SIFT.create()
    kp_roi, des_roi = sift_roi.detectAndCompute(roi, None)

    sift_img = cv2.xfeatures2d_SIFT.create()
    kp_img, des_img = sift_img.detectAndCompute(img, None)

    MIN_MATCH_COUNT = 4
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_roi, des_img, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_roi[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, d = roi.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        if M is not None:
            dst = cv2.perspectiveTransform(pts, M)
            if include_steps:
                img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Could not find homography")
            matchesMask = None
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    warped = None
    if matchesMask:
        img_h, img_w, _ = img.shape
        warped = cv2.warpPerspective(src=roi, M=M, dsize=(img_w, img_h))

    out = cv2.drawMatches(roi, kp_roi, img, kp_img, good, None, **draw_params)

    return warped, out


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
            hstack1 = show_on_row(blur, th)
            hstack2 = show_on_row(morph, output)
            output = show_on_col(hstack1, hstack2)
        else:
            output = result
        return output
