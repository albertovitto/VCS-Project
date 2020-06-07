import os
import cv2
import numpy as np

from estensi.painting_retrieval.utils import elliptical_mask, create_features_db


class PaintingRetrieval:

    def __init__(self, db_dir_path, files_dir_path):
        self.db_dir_path = db_dir_path
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.knn = cv2.ml.KNearest_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # load files or create if missed
        try:
            self.features_db = np.load(os.path.join(files_dir_path, 'features_db.npy'))
            self.img_features_db = np.load(os.path.join(files_dir_path, 'img_features_db.npy'))
        except IOError:
            print("Creating features db ...")
            self.features_db, self.img_features_db = create_features_db(db_dir_path, files_dir_path)
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
        if dsc is not None:
            for d in dsc:
                ret, _, _, _ = self.knn.findNearest(d.reshape((1, len(d))), 1)
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
            rank0_img = cv2.imread(os.path.join(self.db_dir_path, "{:03d}.png".format(rank_keys[0])))
            gray_rank0_img = cv2.cvtColor(rank0_img, cv2.COLOR_BGR2GRAY)
            kp_rank0, dsc_rank0 = self.sift.detectAndCompute(gray_rank0_img, None)

            matches = self.flann.knnMatch(dsc, dsc_rank0, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            if len(good) < 10:
                rank_keys.insert(0, -1)
                rank_values.insert(0, -1)

        return rank_keys, rank_values
