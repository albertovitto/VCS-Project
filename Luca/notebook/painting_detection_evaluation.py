import time

import numpy as np
import cv2

from Luca.vcsp.painting_detection.evaluation import create_random_dict_for_test_set, read_dict_for_test_set, \
    create_test_set, eval_test_set, learn_best_detection


if __name__ == '__main__':

    # RUN ONCE
    # d = create_random_dict_for_test_set()
    # print(d)
    # copy d in read_dict_test_set

    # dict = read_dict_for_test_set()
    # create_test_set(dict)

    """start_time = time.time()
    avg_precision, avg_recall = eval_test_set(include_partial_results=True)
    print("precision = {}, recall = {}".format(avg_precision, avg_recall))
    print("{} seconds".format(time.time() - start_time))"""

    """param_grid = {
        "MIN_ROTATED_BOX_AREA_PERCENT": [0.7, 0.8, 0.9], 
        "MIN_ROTATED_ELLIPSE_AREA_PERCENT": [0.5, 0.6, 0.7],
        "MIN_POLY_AREA_PERCENT": [0.5, 0.6, 0.7, 0.9]
    }"""

    param_grid = {
        "MIN_HULL_AREA_PERCENT_OF_IMG": [0.01, 0.02, 0.03],
        "MIN_VARIANCE": [11, 25],
        "MIN_HULL_AREA_PERCENT_OF_MAX_HULL": [0.1, 0.15]
    }

    learn_best_detection(param_grid=param_grid)
