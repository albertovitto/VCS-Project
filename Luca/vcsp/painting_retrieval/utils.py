import os
import copy

import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC, SVC, SVR, LinearSVR
from scipy.spatial.distance import euclidean, minkowski


# Used for BOW
def create_vocabulary_db(db_dir_name):
    imgs_path_list = []
    for filename in os.listdir(db_dir_name):
        imgs_path_list.append(db_dir_name + filename)

    sift = cv2.xfeatures2d.SIFT_create()

    vocabulary_size = 20000
    bow = cv2.BOWKMeansTrainer(vocabulary_size)

    for p in imgs_path_list:
        img = cv2.imread(p)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, dsc = sift.detectAndCompute(gray_img, None)
        bow.add(dsc)

    # print(bow.descriptorsCount())
    # len(features) = 169869
    # 169869 / 95 = 1788 per img

    print("Clustering vocabulary ...")
    vocabulary = bow.cluster()
    np.save('../../dataset/vocabulary.npy', vocabulary)

    return vocabulary


# Used for BOW
def read_vocabulary_db():
    vocabulary = np.load('../../dataset/vocabulary.npy')
    return vocabulary


# Used for BOW
def read_features_db():
    features = np.load('../../dataset/features.npy')
    return features


# Used for BOW
def extract_features_db(db_dir_name):
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.BFMatcher()
    bow = cv2.BOWImgDescriptorExtractor(sift, matcher)
    bow.setVocabulary(read_vocabulary_db())

    imgs_path_list = []
    for filename in os.listdir(db_dir_name):
        imgs_path_list.append(db_dir_name + filename)

    features = []
    for p in imgs_path_list:
        img = cv2.imread(p)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp = sift.detect(gray_img)
        f = bow.compute(gray_img, kp)
        features.extend(f)

    np.save('../../dataset/features.npy', features)

    return features


# Used for BOW
def retrieve_img(img):
    # leggi le features_db
    features_db = read_features_db()

    # estrai la feature_img
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.BFMatcher()
    bow = cv2.BOWImgDescriptorExtractor(sift, matcher)
    bow.setVocabulary(read_vocabulary_db())

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape
    w_perc = int(w * 0.25)
    h_perc = int(h * 0.25)
    gray_img = gray_img[0 + h_perc:h - h_perc, 0 + w_perc:w - w_perc]

    cv2.imshow("crop", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    kp = sift.detect(gray_img)

    output = copy.deepcopy(gray_img)
    output = cv2.drawKeypoints(output, kp, None)
    cv2.imshow("crop", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    feature_img = bow.compute(gray_img, kp)
    np.save('../../dataset/feature.npy', feature_img)

    # seleziona la feature_db più simile a feature_img
    """knn = cv2.ml.KNearest_create()
    knn.train(features_db, cv2.ml.ROW_SAMPLE, np.arange(len(features_db)))
    ret, results, neighbours, dist = knn.findNearest(feature_img, 6)
    print(ret)
    print(results)
    print(neighbours)
    print(dist)"""

    neighbour = NearestNeighbors(n_neighbors=5)
    neighbour.fit(features_db)
    dist, result = neighbour.kneighbors(feature_img)
    print(dist)
    print(result)

    """svm = LinearSVC()
    svm.fit(features_db, np.arange(len(features_db)))
    predict = svm.predict(feature_img)
    print(predict)"""


def FORMER_create_all_features_db(db_dir_name):
    imgs_path_list = []
    for filename in os.listdir(db_dir_name):
        imgs_path_list.append(db_dir_name + filename)

    sift = cv2.xfeatures2d.SIFT_create()

    features_db = []
    features_range = []
    features_range.append(0)

    len_dsc_prev = 0
    for i, p in enumerate(imgs_path_list):
        img = cv2.imread(p)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, dsc = sift.detectAndCompute(gray_img, None)
        features_db.extend(dsc)
        if i > 0:
            features_range.append(len_dsc_prev + features_range[i - 1])
        len_dsc_prev = len(dsc)

    np.save('../../dataset/all_features_db.npy', features_db)
    np.save('../../dataset/range_features_db.npy', features_range)


# USED THIS
def create_all_features_db(db_dir_path, files_dir_path):
    imgs_path_list = []
    for filename in os.listdir(db_dir_path):
        imgs_path_list.append(db_dir_path + filename)

    sift = cv2.xfeatures2d.SIFT_create()

    features_db = []
    img_features_db = []

    for i, p in enumerate(imgs_path_list):
        img = cv2.imread(p)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, dsc = sift.detectAndCompute(gray_img, None)
        features_db.extend(dsc)

        num_repetition = len(dsc)
        repeated_arr = np.repeat(i, num_repetition)
        img_features_db = np.append(img_features_db, repeated_arr).astype(np.int)

    img_features_db = img_features_db.reshape((1, img_features_db.shape[0]))

    np.save(files_dir_path + 'features_db.npy', features_db)
    np.save(files_dir_path + 'img_features_db.npy', img_features_db)

    return features_db, img_features_db


# UNUSED
def retrieve(test_img_path):
    sift = cv2.xfeatures2d.SIFT_create()

    features_db = np.load('../../dataset/all_features_db.npy')
    features_range = np.load('../../dataset/range_features_db.npy')

    img = cv2.imread(test_img_path)
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


# UNUSED
def get_unrolled_range_arr(range_arr, num_features):
    unrolled_arr = []

    for i in range(len(range_arr)):
        if i != len(range_arr) - 1:
            num_repetition = range_arr[i+1] - range_arr[i]
        else:
            num_repetition = num_features - range_arr[i]
        repeated_arr = np.repeat(i, num_repetition)
        unrolled_arr = np.append(unrolled_arr, repeated_arr).astype(np.int)

    return unrolled_arr.reshape((1, len(unrolled_arr)))


