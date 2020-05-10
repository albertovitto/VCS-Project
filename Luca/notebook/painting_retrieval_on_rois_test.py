import time

import numpy as np
import cv2

from Luca.vcsp.painting_retrieval.retrieval import PaintingRetrieval


if __name__ == '__main__':

    db_dir_path = '../../dataset/paintings_db/'
    files_dir_path = '../../dataset/'

    # SECONDO METODO + CLASSE
    retrieval = PaintingRetrieval(db_dir_path, files_dir_path)
    retrieval.train()

    print("ROI0 - Ground Truth = 9")
    img = cv2.imread("./rois_test/roi0.png")
    rank, _ = retrieval.predict(img)
    print(rank[0])

    print("ROI1 - Ground Truth = 76")
    img = cv2.imread("./rois_test/roi1.png")
    rank, _ = retrieval.predict(img)
    print(rank[0])

    print("ROI2 - Ground Truth = NOT IN DB")
    img = cv2.imread("./rois_test/roi2.png")
    rank, _ = retrieval.predict(img)
    print(rank[0])

    print("ROI3 - Ground Truth = 21")
    img = cv2.imread("./rois_test/roi3.png")
    rank, _ = retrieval.predict(img)
    print(rank[0])

    print("ROI4 - Ground Truth = 45")
    img = cv2.imread("./rois_test/roi4.png")
    rank, _ = retrieval.predict(img)
    print(rank[0])

    print("ROI5 - Ground Truth = 93")
    img = cv2.imread("./rois_test/roi5.png")
    rank, _ = retrieval.predict(img)
    print(rank[0])
