import os
import time

import numpy as np
import cv2

from estensi.painting_detection.constants import conf
from estensi.painting_detection.detection import get_bb
from estensi.painting_retrieval.retrieval import PaintingRetrieval
from estensi.utils import get_ground_truth_bbs, bb_iou


RETRIEVAL = "retrieval"
CLASSIFICATION = "classification"
eval_mode = [RETRIEVAL, CLASSIFICATION]


def eval_test_set(test_set_dir_path, ground_truth_set_dir_path, db_dir_path, files_dir_path, mode=RETRIEVAL, rank_scope=5, iou_threshold=0.5, params=conf, verbose=False):
    assert mode in eval_mode
    assert 1 <= rank_scope <= 95

    if not os.path.isdir(test_set_dir_path):
        print("Test set has not been created yet.")
        return

    if mode == CLASSIFICATION:
        rank_scope = 1
        use_extra_check = True
    else:
        use_extra_check = False

    test_set_dict = {}
    for filename in os.listdir(test_set_dir_path):
        video_index, frame_index = filename.split("_", 1)
        frame_index, _ = frame_index.split(".", 1)
        if video_index in test_set_dict:
            test_set_dict[video_index].append(frame_index)
        else:
            test_set_dict[video_index] = [frame_index]

    retrieval = PaintingRetrieval(db_dir_path=db_dir_path, files_dir_path=files_dir_path)
    retrieval.train()

    if verbose:
        print("Testing in {} mode [rank_scope={}, iou_th={}]...".format(mode, rank_scope, iou_threshold))

    start_time = time.time()

    results = {}
    for video in test_set_dict.keys():
        results[video] = {"precision": 0, "count": 0}
        video_test_set = test_set_dict[video]

        for frame in video_test_set:
            img = cv2.imread(os.path.join(test_set_dir_path, "{}_{}.png".format(video, frame)))
            _, rois, painting_bbs = get_bb(img, params=params, include_steps=False)

            gt_bbs = get_ground_truth_bbs(ground_truth_set_dir_path=ground_truth_set_dir_path, video=video, frame=frame)
            gt_painting_bbs = gt_bbs["paintings"]

            should_continue = False
            if mode == RETRIEVAL:
                for painting in gt_painting_bbs:
                    if painting["label"] != -1:
                        should_continue = True
            else:
                should_continue = True

            if should_continue:
                for i, bb in enumerate(painting_bbs):
                    iou_scores = []
                    for gt_bb in gt_painting_bbs:
                        iou = bb_iou(bb, gt_bb["bb"])
                        iou_scores.append(iou)

                    if len(iou_scores) != 0 and np.max(iou_scores) >= iou_threshold:
                        rank, _ = retrieval.predict(rois[i], use_extra_check=use_extra_check)
                        gt_label = gt_painting_bbs[np.argmax(iou_scores)]["label"]
                        if mode == RETRIEVAL and gt_label == -1:
                            pass
                        elif gt_label in rank[:rank_scope]:
                            position = rank[:rank_scope].index(gt_label) + 1
                            results[video]["precision"] += 1 / position
                            results[video]["count"] += 1
                        else:
                            results[video]["precision"] += 0
                            results[video]["count"] += 1

        if results[video]["count"] != 0:
            video_avg_precision = results[video]["precision"] / results[video]["count"]
        else:
            results.pop(video, None)
            video_avg_precision = -1

        if verbose:
            print("--- video {} ---".format(video))
            print("p = {:.2f}".format(video_avg_precision))
            print("-----------------")

    mean_avg_precision = 0
    for video in results:
        video_avg_precision = results[video]["precision"] / results[video]["count"]
        mean_avg_precision += video_avg_precision
    mean_avg_precision = mean_avg_precision / len(results.keys())

    if verbose:
        print("{:.2f} s".format(time.time() - start_time))

    return mean_avg_precision
