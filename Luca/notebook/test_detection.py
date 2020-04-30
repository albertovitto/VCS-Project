import numpy as np
import cv2
from Luca.vcsp.detection import detect_painting


def main():
    video_name = "VIRB0393.MP4"
    #video_name = "GOPR5826.MP4"
    #video_name = "GOPR2045.MP4"
    #video_name = "IMG_4086.MOV"
    #video_name = "GOPR2051.MP4"
    #video_name = "IMG_3803.MOV"

        video_path = "../../%s" % video_name
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error")

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            roi = detect_painting(frame, include_steps=True)
            cv2.imshow("Painting detection", roi)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1)

        else:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


