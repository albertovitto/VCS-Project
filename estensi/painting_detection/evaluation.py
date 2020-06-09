import copy
import os
import time
import itertools

import cv2
import numpy as np

from estensi.painting_detection.constants import conf
from estensi.painting_detection.detection import get_bb
from estensi.utils import get_ground_truth_bbs, bb_iou, f1_score


def get_test_set_dict(dataset_dir_path):
    d = {
        '000': os.path.join(dataset_dir_path, "videos", "000", "VIRB0393.MP4"),
        '001': os.path.join(dataset_dir_path, "videos", "001", "GOPR5825.MP4"),
        '002': os.path.join(dataset_dir_path, "videos", "002", "20180206_114720.mp4"),
        '003': os.path.join(dataset_dir_path, "videos", "003", "GOPR1929.MP4"),
        '004': os.path.join(dataset_dir_path, "videos", "004", "IMG_3803.MOV"),
        '005': os.path.join(dataset_dir_path, "videos", "005", "GOPR2051.MP4"),
        '006': os.path.join(dataset_dir_path, "videos", "006", "IMG_9629.MOV"),
        '007': os.path.join(dataset_dir_path, "videos", "007", "IMG_7852.MOV"),
        '008': os.path.join(dataset_dir_path, "videos", "008", "VIRB0420.MP4"),
        '009': os.path.join(dataset_dir_path, "videos", "009", "IMG_2659.MOV"),
        '010': os.path.join(dataset_dir_path, "videos", "010", "VID_20180529_112706.mp4"),
        '012': os.path.join(dataset_dir_path, "videos", "012", "IMG_4087.MOV"),
        '013': os.path.join(dataset_dir_path, "videos", "013", "20180529_112417_ok.mp4"),
        '014': os.path.join(dataset_dir_path, "videos", "014", "VID_20180529_113001.mp4")
    }

    return d


def create_test_set(test_set_dict, test_set_dir_path):
    if not os.path.isdir(test_set_dir_path):
        os.mkdir(test_set_dir_path)

    if len(os.listdir(test_set_dir_path)) != 0:
        print("Test set already created.")
        return

    print("Creating test set ...")
    for folder, video_path in test_set_dict.items():
        print("Extracting frames for video {} ...".format(folder))
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print("Error: video not opened correctly")

        counter = 0
        pos_frames = 0
        lost_frames = 0
        while video.isOpened():
            ret, frame = video.read()

            if ret:
                frame_name = "{}_{}.png".format(folder, counter)
                cv2.imwrite(os.path.join(test_set_dir_path, frame_name), frame)

                counter += 1

                pos_frames += video.get(cv2.CAP_PROP_FPS)
                if pos_frames > video.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
                video.set(cv2.CAP_PROP_POS_FRAMES, pos_frames)
            else:
                lost_frames += 1
                if lost_frames > 10:
                    break

        print("Extracting frames for video {} Done.".format(folder))

    print("Creating test set ... Done.")


def eval_test_set(test_set_dir_path, ground_truth_set_dir_path, iou_threshold=0.5, params=conf, verbose=False):
    if not os.path.isdir(test_set_dir_path):
        print("Test set has not been created yet.")
        return

    test_set_dict = {}
    for filename in os.listdir(test_set_dir_path):
        video_index, frame_index = filename.split("_", 1)
        frame_index, _ = frame_index.split(".", 1)
        if video_index in test_set_dict:
            test_set_dict[video_index].append(frame_index)
        else:
            test_set_dict[video_index] = [frame_index]

    results = {}
    avg_precision = 0
    avg_recall = 0

    start_time = time.time()

    for video in test_set_dict.keys():
        results[video] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        video_test_set = test_set_dict[video]

        for frame in video_test_set:
            img = cv2.imread(os.path.join(test_set_dir_path, "{}_{}.png".format(video, frame)))
            _, _, painting_bbs = get_bb(img, params=params, include_steps=False)

            gt_bbs = get_ground_truth_bbs(ground_truth_set_dir_path=ground_truth_set_dir_path, video=video, frame=frame)
            gt_painting_bbs = gt_bbs["paintings"]

            for bb in painting_bbs:
                iou_scores = []
                for gt_bb in gt_painting_bbs:
                    iou = bb_iou(bb, gt_bb["bb"])
                    iou_scores.append(iou)

                if len(iou_scores) != 0 and np.max(iou_scores) >= iou_threshold:
                    results[video]['TP'] += 1
                    gt_painting_bbs.pop(np.argmax(iou_scores))
                else:
                    results[video]['FP'] += 1

            results[video]['FN'] += len(gt_painting_bbs)

        TP, FP, FN, TN = results[video].values()

        precision = 1
        if FP != 0:
            precision = TP / (TP + FP)

        recall = 1
        if FN != 0:
            recall = TP / (TP + FN)

        avg_precision += precision
        avg_recall += recall

        if verbose:
            print("------------ video {} ------------".format(video))
            print("TP = {}, FP = {}, FN = {}, TN = {}".format(TP, FP, FN, TN))
            print("p = {:.2f}, r = {:.2f}".format(precision, recall))
            print("-----------------------------------")

    avg_precision = avg_precision / len(test_set_dict.keys())
    avg_recall = avg_recall / len(test_set_dict.keys())
    f1 = f1_score(avg_precision, avg_recall)

    if verbose:
        print("{:.2f} s".format(time.time() - start_time))

    return f1, avg_precision, avg_recall


def hyperparameters_gridsearch(test_set_dir_path, ground_truth_set_dir_path, param_grid):
    keys, values = zip(*param_grid.items())

    hyperparameters_list = []
    for v in itertools.product(*values):
        hyperparameters = dict(zip(keys, v))
        hyperparameters_list.append(hyperparameters)

    start_time = time.time()
    for i in hyperparameters_list:
        print("Testing configuration {} ...".format(i))
        params = copy.deepcopy(conf)
        for key in i.keys():
            params[key] = i[key]

        f1, avg_precision, avg_recall = eval_test_set(test_set_dir_path=test_set_dir_path,
                                                      ground_truth_set_dir_path=ground_truth_set_dir_path,
                                                      params=params)
        print("f1 = {:.2f}, p = {:.2f}, r = {:.2f} for configuration {}".format(f1, avg_precision, avg_recall, i))

    print("--- {:.2f} sec ---".format(time.time() - start_time))