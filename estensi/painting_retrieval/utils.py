import os
import cv2
import numpy as np


def rectangular_mask(img):
    assert len(img.shape) == 2
    h, w = img.shape
    mask = np.zeros((h, w), np.uint8)
    w_perc = int(w * 0.15)
    h_perc = int(h * 0.20)
    cv2.rectangle(mask, (w_perc, h_perc), (w - w_perc, h - h_perc), 255, -1)
    return mask


def elliptical_mask(img):
    assert len(img.shape) == 2
    h, w = img.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(w/2), int(h/2)), (int(w/2 - w/2*0.3), int(h/2 - h/2*0.3)), 0, 0, 360, 255, thickness=-1)
    return mask


def adaptive_mask(img):
    assert len(img.shape) == 2
    h, w = img.shape
    mask = np.zeros((h, w), np.uint8)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 5)
    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((5, 5)), iterations=1)
    _, contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        candidate_hulls = []

        for i, c in enumerate(contours):
            hull = cv2.convexHull(c)
            candidate_hulls.append(hull.reshape(hull.shape[0], hull.shape[2]))

        hull = sorted(candidate_hulls, key=cv2.contourArea, reverse=True)[0]

    cv2.drawContours(mask, [hull], 0, 255, -1)

    mask[0:2, :] = 0
    mask[h - 2:h, :] = 0
    mask[:, 0:2] = 0
    mask[:, w - 2:w] = 0

    num_iter_erosion = int(max(h, w) / 100)
    num_iter_erosion = num_iter_erosion * max(int(max(h, w) / 500), 1)
    num_iter_erosion = num_iter_erosion + 2
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((15, 15)), iterations=num_iter_erosion)
    return mask


def create_features_db(db_dir_path, files_dir_path):
    imgs_path_list = []
    for filename in os.listdir(db_dir_path):
        imgs_path_list.append(os.path.join(db_dir_path, filename))

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

    np.save(os.path.join(files_dir_path, 'features_db.npy'), features_db)
    np.save(os.path.join(files_dir_path, 'img_features_db.npy'), img_features_db)

    features_db = np.load(os.path.join(files_dir_path, 'features_db.npy'))
    img_features_db = np.load(os.path.join(files_dir_path, 'img_features_db.npy'))

    return features_db, img_features_db
