import cv2
import numpy as np


def sift_feature_matching_and_homography(roi, img, include_steps=True):
    # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    sift_roi = cv2.xfeatures2d_SIFT.create()
    kp_roi, des_roi = sift_roi.detectAndCompute(roi, None)
    out_roi = cv2.drawKeypoints(roi, kp_roi, None)

    sift_img = cv2.xfeatures2d_SIFT.create()
    kp_img, des_img = sift_img.detectAndCompute(img, None)
    out_img = cv2.drawKeypoints(img, kp_img, None)

    if include_steps:
        cv2.imshow('ROI', roi)
        cv2.imshow('ROI + SIFT', out_roi)
        cv2.imshow('IMG', img)
        cv2.imshow('IMG + SIFT', out_img)
        cv2.waitKey(-1)

    MIN_MATCH_COUNT = 10
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
        dst = cv2.perspectiveTransform(pts, M)
        img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    if matchesMask:
        img_h, img_w, _ = img.shape
        warped = cv2.warpPerspective(src=roi, M=M, dsize=(img_w, img_h))
        cv2.imshow('WARPED', warped)

    out = cv2.drawMatches(roi, kp_roi, img, kp_img, good, None, **draw_params)
    cv2.imshow('OUT', out)
    cv2.waitKey(-1)


if __name__ == '__main__':
    # roi = cv2.imread('../roi/out4.jpg')
    # img = cv2.imread('../../dataset/paintings_db/076.png')
    # roi = cv2.imread('../roi/out1.jpg')
    # img = cv2.imread('../../dataset/paintings_db/021.png')
    roi = cv2.imread('../roi/out0.jpg')
    img = cv2.imread('../../dataset/paintings_db/045.png')
    sift_feature_matching_and_homography(roi, img)
