import os
import numpy as np
import cv2

from Luca.vcsp.painting_detection.detection import get_bb
from Luca.vcsp.painting_detection.evaluation import bb_iou, get_ground_truth_bbs

from Luca.vcsp.painting_detection.constants import conf
from Luca.vcsp.painting_retrieval.retrieval import PaintingRetrieval

VIDEOS_FOLDER_PATH = os.path.join('..', '..', 'dataset', 'videos')
TEST_SET_FOLDER_PATH = os.path.join('..', '..', 'dataset', 'painting_detection_test_set')
GROUND_TRUTH_FOLDER_PATH = os.path.join('..', '..', 'dataset', 'ground_truth')
DATASET_FOLDER_PATH = os.path.join("..", "..", "dataset")
DB_FOLDER_PATH = os.path.join("..", "..", "dataset", "paintings_db")


def eval_test_set(iou_threshold=0.5, rank_scope=5, params=conf, verbose=False):
    test_set_dict = {}
    for filename in os.listdir(TEST_SET_FOLDER_PATH):
        video_index, frame_index = filename.split("_", 1)
        frame_index, _ = frame_index.split(".", 1)
        if video_index in test_set_dict:
            test_set_dict[video_index].append(frame_index)
        else:
            test_set_dict[video_index] = [frame_index]

    retrieval = PaintingRetrieval(db_dir_path=DB_FOLDER_PATH, files_dir_path=DATASET_FOLDER_PATH)
    retrieval.train()

    results = {}
    for video in test_set_dict.keys():
        results[video] = {"precision": 0, "count": 0}
        video_test_set = test_set_dict[video]

        for frame in video_test_set:
            img = cv2.imread(os.path.join(TEST_SET_FOLDER_PATH, "{}_{}.png".format(video, frame)))
            _, rois, painting_bbs = get_bb(img, params=params, include_steps=False)

            gt_bbs = get_ground_truth_bbs(video, frame)
            gt_painting_bbs = gt_bbs["paintings"]

            for i, bb in enumerate(painting_bbs):
                iou_scores = []
                for gt_bb in gt_painting_bbs:
                    iou = bb_iou(bb, gt_bb["bb"])
                    iou_scores.append(iou)

                if len(iou_scores) != 0 and np.max(iou_scores) >= iou_threshold:
                    rank, _ = retrieval.predict(rois[i])
                    gt_label = gt_painting_bbs[np.argmax(iou_scores)]["label"]
                    if gt_label == -1:
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

    return mean_avg_precision
