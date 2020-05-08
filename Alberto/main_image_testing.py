import numpy as np
import cv2
import random
from painting_detection import painting_detection_refine
from painting_rectification import painting_rectification, test_rectify
from utils import resize_image


def main():
    # 000/VIRB0400.MP4  000/VIRB0406.MP4 720  002/20180206_114720.mp4
    # 005/GOPR2043.MP4 1080  011/3.mp4 2k
    # video_path = "dataset/videos/000/VIRB0400.MP4"
    # video_path = "dataset/videos/000/VIRB0406.MP4"
    # video_path = "dataset/videos/002/20180206_114720.mp4"
    # video_path = "dataset/videos/003/GOPR1940.MP4"
    # video_path = "dataset/videos/005/GOPR2043.MP4"
    # video_path = "dataset/videos/003/GOPR1940.MP4"
    # video_path = "dataset/videos/011/3.mp4"
    image = cv2.imread(
        "Alberto/Detection refinement #0_screenshot_07.05.2020.png", cv2.IMREAD_GRAYSCALE)
    show, warped = test_rectify(image)
    if show.any():
        cv2.imshow("Painting detection results:", show)
    if warped.any():
        cv2.imshow("fdfd", warped)
    key = cv2.waitKey(0)


if __name__ == "__main__":
    main()
