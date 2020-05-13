import cv2
import numpy as np
from pandas import read_csv
from Luca.vcsp.painting_retrieval.utils import get_unrolled_range_arr


def get_painting_info_from_csv(filename):
    df = read_csv("../dataset/data.csv")
    row = df.loc[df['Image'] == filename]
    return row['Title'].values[0], row['Author'].values[0], row['Room'].values[0]


def sift_feature_matching_and_homography(roi, img, include_steps=False):
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

    warped = None
    if matchesMask:
        img_h, img_w, _ = img.shape
        warped = cv2.warpPerspective(src=roi, M=M, dsize=(img_w, img_h))
        # cv2.imshow('WARPED', warped)

    out = cv2.drawMatches(roi, kp_roi, img, kp_img, good, None, **draw_params)
    return warped, out
    # cv2.imshow('OUT', out)
    # cv2.waitKey(-1)


def retrieve_img_brute_force(roi):
    sift_roi = cv2.xfeatures2d_SIFT.create()
    kp_roi, des_roi = sift_roi.detectAndCompute(roi, None)
    out_roi = cv2.drawKeypoints(roi, kp_roi, None)

    best_good = []
    best = None
    MIN_MATCH_COUNT = 10
    for i in range(95):
        filename = "{:03d}.png".format(i)
        print("Processing " + filename)
        img = cv2.imread("../../dataset/paintings_db/{}".format(filename))

        sift_img = cv2.xfeatures2d_SIFT.create()
        kp_img, des_img = sift_img.detectAndCompute(img, None)

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
        if len(good) > len(best_good):
            best = i
            best_good = good

    if len(best_good) > MIN_MATCH_COUNT:
        filename = "{:03d}.png".format(best)
        print("Processing " + filename)
        img = cv2.imread("../../dataset/paintings_db/{}".format(filename))
        sift_img = cv2.xfeatures2d_SIFT.create()
        kp_img, des_img = sift_img.detectAndCompute(img, None)

        src_pts = np.float32([kp_roi[m.queryIdx].pt for m in best_good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_img[m.trainIdx].pt for m in best_good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, d = roi.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(best_good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    if matchesMask:
        img_h, img_w, _ = img.shape
        warped = cv2.warpPerspective(src=roi, M=M, dsize=(img_w, img_h))
        cv2.imshow('WARPED', warped)

    out = cv2.drawMatches(roi, kp_roi, img, kp_img, best_good, None, **draw_params)
    cv2.imshow('OUT', out)
    cv2.waitKey(-1)


def retrieve(img):
    sift = cv2.xfeatures2d.SIFT_create()

    features_db = np.load('../dataset/all_features_db.npy')
    features_range = np.load('../dataset/range_features_db.npy')

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape
    w_perc = int(w * 0.25)
    h_perc = int(h * 0.35)
    gray_img = gray_img[0 + h_perc:h - h_perc, 0 + w_perc:w - w_perc]
    kp, dsc = sift.detectAndCompute(gray_img, None)

    knn = cv2.ml.KNearest_create()
    knn.train(features_db, cv2.ml.ROW_SAMPLE, get_unrolled_range_arr(features_range, len(features_db)))

    results = []
    for d in dsc:
        ret, _, _, _ = knn.findNearest(d.reshape((1, len(d))), 1)
        results.append(int(ret))

    most_freq = np.bincount(results).argmax()

    return features_db, features_range, results, most_freq


def extract_features(path, bowDiction, sift):
    image = cv2.imread(path)
    return bowDiction.compute(image, sift.detect(image))


def get_BOW():
    # https://github.com/briansrls/SIFTBOW/blob/master/SIFTBOW.py
    sift = cv2.xfeatures2d_SIFT.create()
    descriptors_unclustered = []
    dictionarySize = 5
    BOW = cv2.BOWKMeansTrainer(dictionarySize)
    for i in range(95):
        filename = "{:03d}.png".format(i)
        image = cv2.imread("../../dataset/paintings_db/{}".format(filename))
        print("Processing " + filename)
        kp, dsc = sift.detectAndCompute(image, None)
        BOW.add(dsc)

    dictionary = BOW.cluster()

    return dictionary

    return bowDiction


def get_features_db(bowDiction, sift):
    train_desc = []
    train_labels = []
    for i in range(95):
        filename = "{:03d}.png".format(i)
        print("Extracting features for " + filename)
        path = "../../dataset/paintings_db/{}".format(filename)
        train_desc.extend(extract_features(path, bowDiction, sift))
        train_labels.append(i)
    return train_desc, train_labels


if __name__ == '__main__':
    roi = cv2.imread('../roi/out4.jpg')
    img = cv2.imread('../../dataset/paintings_db/076.png')

    # roi = cv2.imread('../roi/out1.jpg')
    # img = cv2.imread('../../dataset/paintings_db/021.png')

    # roi = cv2.imread('../roi/out0.jpg')
    # img = cv2.imread('../../dataset/paintings_db/045.png')

    sift_feature_matching_and_homography(roi, img)

    # retrieve_img_brute_force(roi)

    sift = cv2.xfeatures2d.SIFT_create()
    # dictionary = get_BOW()
    # np.save("dictionary.npy", dictionary)
    # dictionary = np.load("dictionary.npy")
    dictionary = np.load("../../dataset/vocabulary.npy")

    bowDiction = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    bowDiction.setVocabulary(dictionary)
    print("bow dictionary: {}".format(np.shape(dictionary)))

    train_desc, train_labels = get_features_db(bowDiction, sift)
    np.save("train_desc.npy", train_desc)
    np.save("train_labels.npy", train_labels)
    # train_desc = np.load("train_desc.npy")
    # train_labels = np.load("train_labels.npy")

    print("svm items {} {}".format(len(train_desc), len(train_desc[0])))
    train_desc = np.array(train_desc)
    train_labels = np.array(train_labels)
    svm = cv2.ml.SVM_create()
    svm.train(train_desc, cv2.ml.ROW_SAMPLE, np.arange(95))

    path = '../roi/out4.jpg'
    roi = cv2.imread(path)
    f = extract_features(path, bowDiction, sift)
    p = svm.predict(f)
    print("GT: 76 - P: {}".format(p))
