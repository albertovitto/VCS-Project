import copy
import os
import time
import itertools

import numpy as np
import cv2
from sklearn.externals.joblib import parallel_backend, Parallel, delayed

from Luca.vcsp.painting_detection.detection import get_bb
from Luca.vcsp.painting_detection.constants import conf


VIDEOS_FOLDER_PATH = os.path.join('..', '..', 'dataset', 'videos')
TEST_SET_FOLDER_PATH = os.path.join('..', '..', 'dataset', 'painting_detection_test_set')
GROUND_TRUTH_FOLDER_PATH = os.path.join('..', '..', 'dataset', 'painting_detection_ground_truth')


# RUN ONCE
def create_random_dict_for_test_set():

    d = {}
    for video_folder in os.listdir(VIDEOS_FOLDER_PATH):
        video_folder_path = os.path.join(VIDEOS_FOLDER_PATH, video_folder)
        videos_list = os.listdir(video_folder_path)
        d[video_folder] = os.path.join(video_folder_path, videos_list[np.random.randint(len(videos_list))])

    return d


def read_dict_for_test_set():
    d = {'000': '..\\..\\dataset\\videos\\000\\VIRB0393.MP4',
         '001': '..\\..\\dataset\\videos\\001\\GOPR5825.MP4',
         '002': '..\\..\\dataset\\videos\\002\\20180206_114720.mp4',
         '003': '..\\..\\dataset\\videos\\003\\GOPR1929.MP4',
         '004': '..\\..\\dataset\\videos\\004\\IMG_3803.MOV',
         '005': '..\\..\\dataset\\videos\\005\\GOPR2051.MP4',
         '006': '..\\..\\dataset\\videos\\006\\IMG_9629.MOV',
         '007': '..\\..\\dataset\\videos\\007\\IMG_7852.MOV',
         '008': '..\\..\\dataset\\videos\\008\\VIRB0420.MP4',
         '009': '..\\..\\dataset\\videos\\009\\IMG_2659.MOV',
         '010': '..\\..\\dataset\\videos\\010\\VID_20180529_112706.mp4',
         '011': '..\\..\\dataset\\videos\\011\\3.mp4',
         '012': '..\\..\\dataset\\videos\\012\\IMG_4087.MOV',
         '013': '..\\..\\dataset\\videos\\013\\20180529_112417_ok.mp4',
         '014': '..\\..\\dataset\\videos\\014\\VID_20180529_113001.mp4'
         }

    return d


def create_test_set(videos_path_list):

    if not os.path.isdir(TEST_SET_FOLDER_PATH):
        os.mkdir(TEST_SET_FOLDER_PATH)

    if len(os.listdir(TEST_SET_FOLDER_PATH)) != 0:
        print("Test set already created.")
        return

    print("Creating test set ...")
    for folder, video_path in videos_path_list.items():
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
                cv2.imwrite(os.path.join(TEST_SET_FOLDER_PATH, frame_name), frame)

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


def bb_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = abs(max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def get_ground_truth_bbs(video_name, frame_name, test_img_shape):
    # 0 = painting
    # 1 = person
    # 2 = statue
    bbs = {0: [], 1: [], 2: []}

    try:
        file = open(os.path.join(GROUND_TRUTH_FOLDER_PATH, "{}_{}.txt".format(video_name, frame_name)), "r")
    except FileNotFoundError:
        # print("Ground truth file not found.")
        return bbs

    lines = file.readlines()
    for line in lines:
        chunks = line.split(" ")
        yolo_record = (int(chunks[0]), float(chunks[1]), float(chunks[2]), float(chunks[3]), float(chunks[4]))
        decoded_bb = yolo_format_decoder(yolo_record, test_img_shape)
        bbs[decoded_bb[0]].append(decoded_bb[1:])

    file.close()

    return bbs


def yolo_format_decoder(record, img_shape):
    cls = record[0]
    center_x = record[1]
    center_y = record[2]
    w = record[3]
    h = record[4]

    img_h = img_shape[0]
    img_w = img_shape[1]

    w = int(img_w * w)
    h = int(img_h * h)
    x = int(img_w * center_x - w / 2)
    y = int(img_h * center_y - h / 2)

    return cls, x, y, w, h


def eval_test_set(iou_threshold=0.5, params=conf, include_partial_results=False):

    test_set_dict = {}
    for filename in os.listdir(TEST_SET_FOLDER_PATH):
        video_index, frame_index = filename.split("_", 1)
        frame_index, _ = frame_index.split(".", 1)
        if int(video_index) != 11:  # 011 is a 2K video => computation too slow
            if video_index in test_set_dict:
                test_set_dict[video_index].append(frame_index)
            else:
                test_set_dict[video_index] = [frame_index]

    results = {}
    avg_precision = 0
    avg_recall = 0
    for video in test_set_dict.keys():
        results[video] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        video_test_set = test_set_dict[video]

        for frame in video_test_set:
            img = cv2.imread(os.path.join(TEST_SET_FOLDER_PATH, "{}_{}.png".format(video, frame)))
            _, _, painting_bbs = get_bb(img, params=params, include_steps=False)

            gt_bbs = get_ground_truth_bbs(video, frame, img.shape)
            gt_painting_bbs = gt_bbs[0]

            for bb in painting_bbs:
                iou_scores = []
                for gt_bb in gt_painting_bbs:
                    iou = bb_iou(bb, gt_bb)
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

        if include_partial_results:
            print("--- video {} ---".format(video))
            print("TP FP FN TN")
            print("{} {} {} {}".format(TP, FP, FN, TN))
            print("p = {}, r = {}".format(precision, recall))
            print("-----------------")

    avg_precision = avg_precision / len(test_set_dict.keys())
    avg_recall = avg_recall / len(test_set_dict.keys())

    return avg_precision, avg_recall

    """
    for each IMG in TEST_SET_FOLDER_PATH:

        extract PREDICTED BB with get_bb(IMG)
        extract GOUND TRUTH BB from annotation file

        for each PREDICTED BB:
            for each GROUND TRUTH BB:
                calculate IOU

            extract GROUND TRUTH BB with maximum IOU
            if >= THRESHOLD (0.5):
                found TP
                remove this GROUND TRUTH BB
            else:
                found FP

        if GROUND TRUTH BB is not empty:
            found FN(s)

        calculate PRECISION and RECALL for IMG

    calculate AVERAGE PRECISION and RECALL for test set

    """


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def apply(x):
    print("Testing configuration {} ...".format(x))
    params = copy.deepcopy(conf)
    for key in x.keys():
        params[key] = x[key]

    avg_precision, avg_recall = eval_test_set(params=params)
    f1 = f1_score(avg_precision, avg_recall)
    print("f1 = {:.2f}, p = {:.2f}, r = {:.2f} for configuration {}".format(f1, avg_precision, avg_recall, x))


def learn_best_detection(param_grid):

    keys, values = zip(*param_grid.items())

    hyperparameters_list = []
    for v in itertools.product(*values):
        hyperparameters = dict(zip(keys, v))
        hyperparameters_list.append(hyperparameters)

    start_time = time.time()
    for i in range(0, len(hyperparameters_list), 4):

        with parallel_backend('threading'):
            Parallel()(delayed(apply)(x) for x in hyperparameters_list[i:i+4])

    print("--- {} sec ---".format(time.time() - start_time))



