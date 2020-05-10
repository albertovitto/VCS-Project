import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def build_dictionary(db_dir_name, dictionary_size):
    sift = cv2.xfeatures2d.SIFT_create()
    
    imgs_path_list = []
    for filename in os.listdir(db_dir_name):
        imgs_path_list.append(db_dir_name + filename)

    print('Computing descriptors...')
    desc_list = []
    for p in imgs_path_list:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, dsc = sift.detectAndCompute(gray, None)
        desc_list.extend(dsc)
    print('Done.')

    print('Creating BoW dictionary using K-Means clustering with k={}..'.format(dictionary_size))
    dictionary = MiniBatchKMeans(n_clusters=dictionary_size, batch_size=100, verbose=1)
    dictionary.fit(np.array(desc_list))
    return dictionary


if __name__ == '__main__':
    db_dir_name = '../../dataset/paintings_db/'
    dictionary = build_dictionary(db_dir_name, 500)
