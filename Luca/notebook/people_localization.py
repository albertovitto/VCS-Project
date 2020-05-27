import os

import numpy as np
import cv2

from Luca.vcsp.people_detection.detection import PeopleDetector
from Luca.vcsp.people_localization.utils import highlight_map_room
from Luca.vcsp.utils import horizontal_stack
from Luca.vcsp.utils.multiple_show import show_on_row

if __name__ == '__main__':

    pd = PeopleDetector()

    # video_name = '000/VIRB0393.MP4'
    # video_name = '001/GOPR5826.MP4'
    # video_name = '005/GOPR2045.MP4'
    # video_name = '012/IMG_4086.MOV'
    # video_name = '005/GOPR2051.MP4'
    # video_name = '004/IMG_3803.MOV'  # statua
    # video_name = '008/VIRB0419.MP4'
    # video_name = '008/VIRB0427.MP4'
    # video_name = '012/IMG_4080.MOV'
    video_name = '002/20180206_114720.mp4'

    video_path = '../../dataset/videos/%s' % video_name

    # dict = read_dict_for_test_set()
    # video_path = dict['014']

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: video not opened correctly")

    lost_frames = 0
    pos_frames = 0
    skip_frames = False
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            output = pd.get_bb(frame)
            map_img = highlight_map_room(np.random.randint(1, 22), map_path=os.path.join("..", "..", "dataset", "map.png"))

            hstack = show_on_row(output, map_img)
            cv2.imshow("Frame", hstack)

            key = cv2.waitKey(1)
            if key == ord('q'):  # quit
                break
            if key == ord('p'):  # pause
                cv2.waitKey(-1)

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