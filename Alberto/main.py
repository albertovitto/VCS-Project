import numpy as np
import cv2
import random

from painting_detection import painting_detection, painting_detection_refine
from painting_rectification import painting_rectification
from utils import resize_image


def main():
    # 000/VIRB0400.MP4  000/VIRB0406.MP4 720  002/20180206_114720.mp4
    # 005/GOPR2043.MP4 1080  011/3.mp4 2k

    video_path = "dataset/videos/003/GOPR1929.MP4"
    #video_path = "dataset/videos/000/VIRB0406.MP4"
    #video_path = 'dataset/videos/001/GOPR5826.MP4'
    # video_path = "dataset/videos/002/20180206_114720.mp4"
    # video_path = "dataset/videos/003/GOPR1940.MP4"
    # video_path = "dataset/videos/005/GOPR2043.MP4"
    # video_path = "dataset/videos/011/3.mp4"

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Cant open video")

    lost_frames = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            roi, boxes = painting_detection(frame)
            if roi is not None:
                cv2.imshow("Painting detection results:", roi)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("p"):
                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    window_name = "Detection refinement #%d" % (i)
                    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi = frame_bw[y:y + h, x:x + w]
                    _, cropped_rectangle, refined_box = painting_detection_refine(
                        roi, i)
                    # if refined_box:  # refined box found
                    #     x, y, w, h = refined_box[0]
                    #     roi = frame_bw[y:y + h, x:x + w]
                    # else keep old box

                    undistorted = None

                    if not cropped_rectangle.any():
                        cv2.imshow(window_name, roi)
                        _, undistorted = painting_rectification(roi)
                    else:
                        cv2.imshow(window_name, cropped_rectangle)
                        _, undistorted = painting_rectification(
                            cropped_rectangle)

                    if undistorted[0][0] != -1:
                        cv2.imshow("Rect #%d" % (i), undistorted)

                cv2.waitKey(-1)
                cv2.destroyAllWindows()

        else:
            lost_frames += 1
            print("Lost frames: ", lost_frames)
            if lost_frames > 10:
                print("Too many errors reading video or video ended")
                break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
