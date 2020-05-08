import os

import numpy as np
import cv2


def create_vocabulary_db(db_dir_name):
    imgs_path_list = []
    for filename in os.listdir(db_dir_name):
        imgs_path_list.append(db_dir_name + filename)

    sift = cv2.xfeatures2d.SIFT_create()

    vocabulary_size = 500
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


def read_vocabulary_db():
    vocabulary = np.load('../../dataset/vocabulary.npy')
    return vocabulary


def read_features_db():
    features = np.load('../../dataset/features.npy')
    return features


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


def retrieve_img(img):
    # leggi le features_db
    features_db = read_features_db()

    # estrai la feature_img
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.BFMatcher()
    bow = cv2.BOWImgDescriptorExtractor(sift, matcher)
    bow.setVocabulary(read_vocabulary_db())

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray_img)
    feature_img = bow.compute(gray_img, kp)

    # seleziona la feature_db più simile a feature_img
    knn = cv2.ml.KNearest_create()
    knn.train(features_db, cv2.ml.ROW_SAMPLE, np.arange(len(features_db)))
    ret, results, neighbours, dist = knn.findNearest(feature_img, 6)
    print(ret)
    print(results)
    print(neighbours)
    print(dist)
