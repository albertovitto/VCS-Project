import os

import numpy as np
import cv2
from Luca.vcsp.painting_retrieval.utils import create_all_features_db, rectangular_mask, elliptical_mask, adaptive_mask


class PaintingRetrieval:

    def __init__(self, db_dir_path, files_dir_path):
        self.db_dir_path = db_dir_path
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.knn = cv2.ml.KNearest_create()

        # load files or create if missed
        try:
            self.features_db = np.load(os.path.join(files_dir_path, 'features_db.npy'))
            self.img_features_db = np.load(os.path.join(files_dir_path, 'img_features_db.npy'))
        except IOError:
            print("Creating features db ...")
            self.features_db, self.img_features_db = create_all_features_db(db_dir_path, files_dir_path)
            print("Done.")

    def train(self):
        self.knn.train(self.features_db,
                       cv2.ml.ROW_SAMPLE,
                       self.img_features_db)

    def predict(self, test_img, use_extra_check=False):
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        mask = elliptical_mask(gray_img)
        masked_data = cv2.bitwise_and(gray_img, gray_img, mask=mask)

        kp, dsc = self.sift.detectAndCompute(masked_data, None)

        results = []
        # for each descriptor, find the most similar img_db
        for d in dsc:
            # ret, _, _, _ = self.knn.findNearest(d.reshape((1, len(d))), 1)
            # results.append(int(ret))
            ret, _, _, dist = self.knn.findNearest(d.reshape((1, len(d))), 2)
            if dist[0, 0] < 0.75 * dist[0, 1]:  # Lowe's ratio test
                results.append(int(ret))

        # create ranked list in descending order of similarity
        rank = {}
        num_imgs_db = int(np.max(self.img_features_db, axis=1)) + 1
        for i, v in enumerate(np.bincount(results, minlength=num_imgs_db)):
            rank[i] = v
        rank = {k: v for k, v in sorted(rank.items(), key=lambda item: item[1], reverse=True)}
        rank_keys = list(rank.keys())
        rank_values = list(rank.values())

        if use_extra_check:
            # check on histogram match for rank[0]
            rank0_img = cv2.imread(os.path.join(self.db_dir_path, "{:03d}.png".format(rank_keys[0])))
            gray_rank0_img = cv2.cvtColor(rank0_img, cv2.COLOR_BGR2GRAY)
            gray_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            rank0_hist = cv2.calcHist([gray_rank0_img], [0], None, [256], [0, 256])
            test_hist = cv2.calcHist([gray_test_img], [0], None, [256], [0, 256])
            cv2.normalize(rank0_hist, rank0_hist, norm_type=cv2.NORM_L1)
            cv2.normalize(test_hist, test_hist, norm_type=cv2.NORM_L1)
            # rank0_hist = cv2.calcHist([rank0_img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            # test_hist = cv2.calcHist([test_img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

            ret = cv2.compareHist(rank0_hist, test_hist, method=cv2.HISTCMP_INTERSECT)
            print(ret)
            if ret < 0.5:  # RGB => 2200, GRAY => 50000, NORM + GRAY => 0.5
                rank_keys.insert(0, -1)
                rank_values.insert(0, -1)

        return rank_keys, rank_values
