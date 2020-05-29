import time

from Luca.vcsp.painting_retrieval.evaluation import eval_test_set


if __name__ == '__main__':

    start_time = time.time()
    mAP = eval_test_set(verbose=True)
    print("map = {:.2f}".format(mAP))
    print("{} seconds".format(time.time() - start_time))
