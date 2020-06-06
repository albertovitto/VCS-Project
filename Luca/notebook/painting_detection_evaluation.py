import time

import numpy as np
import cv2

from Luca.vcsp.painting_detection.evaluation import create_random_dict_for_test_set, read_dict_for_test_set, \
    create_test_set, eval_test_set, learn_best_detection, f1_score


if __name__ == '__main__':

    # RUN ONCE
    # d = create_random_dict_for_test_set()
    # print(d)
    # copy d in read_dict_test_set

    # dict = read_dict_for_test_set()
    # create_test_set(dict)

    start_time = time.time()
    avg_precision, avg_recall = eval_test_set(verbose=True)
    f1 = f1_score(avg_precision, avg_recall)
    print("f1 = {:.2f}, p = {:.2f}, r = {:.2f}".format(f1, avg_precision, avg_recall))
    print("{} seconds".format(time.time() - start_time))

    param_grid = {
        "MIN_ROTATED_BOX_AREA_PERCENT": [0.7, 0.8, 0.9], 
        "MIN_ROTATED_ELLIPSE_AREA_PERCENT": [0.5, 0.6, 0.7],
        "MIN_POLY_AREA_PERCENT": [0.5, 0.6, 0.7, 0.9]
    }

    """param_grid = {
        "MIN_ROTATED_BOX_AREA_PERCENT": [0.5, 0.8, 0.9],
        "MIN_ROTATED_ELLIPSE_AREA_PERCENT": [0.4, 0.6],
        "MAX_GRAY_60_PERCENTILE": [170, 200],
        "MIN_VARIANCE": [11, 18],
        "MIN_HULL_AREA_PERCENT_OF_MAX_HULL": [0.08, 0.15],
        "THRESHOLD_BLOCK_SIZE_FACTOR": [50, 80]
    }"""

    param_grid = {
        "MIN_ROTATED_BOX_AREA_PERCENT": [0.7, 0.75, 0.8, 0.85],
        "MIN_ROTATED_ELLIPSE_AREA_PERCENT": [0.7, 0.75, 0.8, 0.85],
    }

    """param_grid = {
        "MIN_HULL_AREA_PERCENT_OF_MAX_HULL": [0.35, 0.4, 0.45, 0.5],
        "MIN_HULL_AREA_PERCENT_OF_IMG": [0, 0.01, 0.05, 0.1]
    }"""

    #learn_best_detection(param_grid=param_grid)

