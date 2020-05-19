import os

import numpy as np
import cv2


VIDEOS_FOLDER_PATH = os.path.join('..', '..', 'dataset', 'videos')
TEST_SET_FOLDER_PATH = os.path.join(
    '..', '..', 'dataset', 'painting_detection_test_set')
GROUND_TRUTH_FOLDER_PATH = os.path.join(
    '..', '..', 'dataset', 'painting_detection_ground_truth')


# RUN ONCE
def create_random_dict_for_test_set():

    d = {}
    for video_folder in os.listdir(VIDEOS_FOLDER_PATH):
        video_folder_path = os.path.join(VIDEOS_FOLDER_PATH, video_folder)
        videos_list = os.listdir(video_folder_path)
        d[video_folder] = os.path.join(
            video_folder_path, videos_list[np.random.randint(len(videos_list))])

    return d


def read_dict_for_test_set():
    d = [("000", "../../dataset/videos/000/VIRB0393.MP4"),
         ("001", "../../dataset/videos/001/GOPR5825.MP4"),
         ("002", "../../dataset/videos/002/20180206_114720.mp4"),
         ("003", "../../dataset/videos/003/GOPR1929.MP4"),
         ("004", "../../dataset/videos/004/IMG_3803.MOV"),
         ("005", "../../dataset/videos/005/GOPR2051.MP4"),
         ("006", "../../dataset/videos/006/IMG_9629.MOV"),
         ("007", "../../dataset/videos/007/IMG_7852.MOV"),
         ("008", "../../dataset/videos/008/VIRB0420.MP4"),
         ("009", "../../dataset/videos/009/IMG_2659.MOV"),
         ("010", "../../dataset/videos/010/VID_20180529_112706.mp4"),
         ("011", "../../dataset/videos/011/3.mp4"),
         ("012", "../../dataset/videos/012/IMG_4087.MOV"),
         ("013", "../../dataset/videos/013/20180529_112417_ok.mp4"),
         ("014", "../../dataset/videos/014/VID_20180529_113001.mp4")]
    return d


def create_test_set(videos_path_list):

    if not os.path.isdir(TEST_SET_FOLDER_PATH):
        os.mkdir(TEST_SET_FOLDER_PATH)

    if len(os.listdir(TEST_SET_FOLDER_PATH)) != 0:
        print("Test set already created.")
        return

    print("Creating test set ...")
    for folder, video_path in videos_path_list:
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
                cv2.imwrite(os.path.join(
                    TEST_SET_FOLDER_PATH, frame_name), frame)

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


def eval_test_set():
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
