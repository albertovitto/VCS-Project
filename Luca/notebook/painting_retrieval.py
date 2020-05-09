import time

import numpy as np
import cv2

from Luca.vcsp.painting_retrieval.utils import create_vocabulary_db, read_vocabulary_db, extract_features_db,\
    read_features_db, retrieve_img, retrieve, create_all_features_db


if __name__ == '__main__':

    db_dir_name = '../../dataset/paintings_db/'

    """start_time = time.time()

    vocabulary = create_vocabulary_db(db_dir_name)

    end_time = time.time()
    print(end_time-start_time)  # 7 min (num clusters = 500)

    # vocabulary = read_vocabulary_db()

    features = extract_features_db(db_dir_name)

    # features = read_features_db()
    print(len(features))  # 95"""


    """print("ROI0 - Ground Truth = 9")
    img = cv2.imread("./rois_test/roi0.png")
    retrieve_img(img)
 
    print("ROI1 - Ground Truth = 76")
    img = cv2.imread("./rois_test/roi1.png")
    retrieve_img(img)

    print("ROI3 - Ground Truth = 21")
    img = cv2.imread("./rois_test/roi3.png")
    retrieve_img(img)

    print("ROI4 - Ground Truth = 45")
    img = cv2.imread("./rois_test/roi4.png")
    retrieve_img(img)

    print("ROI5 - Ground Truth = 93")
    img = cv2.imread("./rois_test/roi5.png")
    retrieve_img(img)

    print("IMG0 - Ground Truth = 0")
    img = cv2.imread("../../dataset/paintings_db/000.png")
    retrieve_img(img)

    print("IMG10 - Ground Truth = 10")
    img = cv2.imread("../../dataset/paintings_db/010.png")
    retrieve_img(img)

    print("IMG55 - Ground Truth = 55")
    img = cv2.imread("../../dataset/paintings_db/055.png")
    retrieve_img(img)"""

    print("Creating features db ...")
    create_all_features_db(db_dir_name)
    print("Done")

    print("Start")
    features_db, features_range, results, ind = retrieve("./rois_test/roi0.png")
    print(results)
    print(ind)

    print("Start")
    features_db, features_range, results, ind = retrieve("./rois_test/roi1.png")
    print(results)
    print(ind)

    print("Start")
    features_db, features_range, results, ind = retrieve("./rois_test/roi2.png")
    print(results)
    print(ind)

    print("Start")
    features_db, features_range, results, ind = retrieve("./rois_test/roi3.png")
    print(results)
    print(ind)

    print("Start")
    features_db, features_range, results, ind = retrieve("./rois_test/roi4.png")
    print(results)
    print(ind)

    print("Start")
    features_db, features_range, results, ind = retrieve("./rois_test/roi5.png")
    print(results)
    print(ind)

