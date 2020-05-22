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

    learn_best_detection()
