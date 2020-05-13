import numpy as np
import cv2
from Luca.vcsp.painting_retrieval.utils import create_all_features_db, rectangular_mask, elliptical_mask, adaptive_mask


class PaintingRetrieval:

    def __init__(self, db_dir_path, files_dir_path):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.knn = cv2.ml.KNearest_create()

        # load files or create if missed
        try:
            self.features_db = np.load(files_dir_path + 'features_db.npy')
            self.img_features_db = np.load(files_dir_path + 'img_features_db.npy')
        except IOError:
            print("Creating features db ...")
            self.features_db, self.img_features_db = create_all_features_db(db_dir_path, files_dir_path)
            print("Done")

    def train(self):
        self.knn.train(self.features_db,
                       cv2.ml.ROW_SAMPLE,
                       self.img_features_db)

    def predict(self, test_img):
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        mask = elliptical_mask(gray_img)
        masked_data = cv2.bitwise_and(gray_img, gray_img, mask=mask)

        kp, dsc = self.sift.detectAndCompute(masked_data, None)

        results = []
        # for each descriptor, find the most similar img_db
        for d in dsc:
            ret, _, _, _ = self.knn.findNearest(d.reshape((1, len(d))), 1)
            results.append(int(ret))

        # create ranked list in descending order of similarity
        rank = {}
        num_imgs_db = int(np.max(self.img_features_db, axis=1)) + 1
        for i, v in enumerate(np.bincount(results, minlength=num_imgs_db)):
            rank[i] = v
        rank = {k: v for k, v in sorted(rank.items(), key=lambda item: item[1], reverse=True)}

        return list(rank.keys()), list(rank.values())
