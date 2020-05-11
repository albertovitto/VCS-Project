import time

import numpy as np
import cv2

from Luca.vcsp.painting_retrieval.retrieval import PaintingRetrieval
from Luca.vcsp.painting_detection.detection import get_bb
from Luca.vcsp.utils.multiple_show import show_on_row


if __name__ == '__main__':

    db_dir_path = '../../dataset/paintings_db/'
    files_dir_path = '../../dataset/'

    retrieval = PaintingRetrieval(db_dir_path, files_dir_path)
    retrieval.train()

    # video_name = '000/VIRB0393.MP4'
    # video_name = '001/GOPR5826.MP4'
    # video_name = '005/GOPR2045.MP4'
    # video_name = '012/IMG_4086.MOV'
    video_name = '005/GOPR2051.MP4'
    # video_name = '004/IMG_3803.MOV'
    # video_name = '008/VIRB0419.MP4'
    # video_name = '008/VIRB0427.MP4'
    # video_name = '012/IMG_4080.MOV'
    # video_name = '002/20180206_114720.mp4'

    video_path = '../../dataset/videos/%s' % video_name

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: video not opened correctly")

    lost_frames = 0
    pos_frames = 0
    skip_frames = True
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            output, rois = get_bb(frame, include_steps=True)
            cv2.imshow("Painting detection", output)

            key = cv2.waitKey(1)
            if key == ord('q'):  # quit
                break
            if key == ord('p'):  # pause
                cv2.waitKey(-1)
            if key == ord('r'):  # show rois with image retrieval
                for i, roi in enumerate(rois):
                    rank, _ = retrieval.predict(roi)
                    cv2.putText(roi, "{}".format(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3,
                                False)
                    print("Roi {} - rank = {}".format(i, rank))
                    ground_truth = cv2.imread('../../dataset/paintings_db/' + "{:03d}.png".format(rank[0]))
                    cv2.imshow("Roi {}".format(i), show_on_row(roi, ground_truth))
                cv2.waitKey(-1)
                for i, roi in enumerate(rois):
                    cv2.destroyWindow("Roi {}".format(i))

            if skip_frames:
                pos_frames += video.get(cv2.CAP_PROP_FPS)
                video.set(cv2.CAP_PROP_POS_FRAMES, pos_frames)
        else:
            lost_frames += 1
            if lost_frames > 10:
                print("Too many errors reading video or video ended")
                break

    video.release()
    cv2.destroyAllWindows()
