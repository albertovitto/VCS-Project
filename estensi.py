## main definitivo per il progetto!
# Prendetevi pure la libertà di modificare quello che vi pare,
# mi sono limitato a riportare lo scheletro.
## Per modificare i parametri di input su PyCharm senza lanciare da linea di comando
# accanto al pulsante play in alto a destra,
# cliccare sulla freccia accanto nome del modulo da avviare.
# Cliccare poi su 'Edit Configurations...'
# Modificare il campo parameters.
# Oppure tramite shortcut: Shift+Alt+F10 seguito da F4

import argparse
import os
import cv2
import numpy as np
from Cristian.YOLOv3.test_yolo import Yolo
import Cristian.image_processing.people_localization as pl
from Cristian.image_processing.people_localization import PeopleLocator
from Cristian.image_processing.retrieval_utils import sift_feature_matching_and_homography
from Luca.vcsp.painting_detection.detection import get_bb
from Luca.vcsp.painting_rectification.rectification import rectify
from Luca.vcsp.painting_retrieval.retrieval import PaintingRetrieval
from Luca.vcsp.utils.drawing import draw_bb
from Luca.vcsp.utils.multiple_show import show_on_row


def arg_parse():
    parser = argparse.ArgumentParser(description='Vision and Cognitive Systems project: Gallerie Estensi')
    parser.add_argument('--video', dest='video', help='path of the selected video', type=str, required=True)
    parser.add_argument('--include_steps', dest='include_steps', action='store_true',
                        help='toggles additional screens that show more info (like intermediate steps)')
    parser.add_argument('--skip_frames', dest='skip_frames', action='store_true', help='toggles frame skip')
    return parser.parse_args()


def main():
    args = arg_parse()

    path = os.path.abspath(os.path.dirname(__file__))
    db_dir_path = os.path.join(path, "dataset", "paintings_db")
    files_dir_path = os.path.join(path, "dataset")

    retrieval = PaintingRetrieval(db_dir_path, files_dir_path)
    retrieval.train()

    yolo = Yolo(yolo_path=os.path.join(path, "Cristian", "YOLOv3"))

    people_locator = PeopleLocator(distance=pl.CENTER_DISTANCE, weighting=pl.SQRT_AREA, voting=True,
                                   verbose=args.include_steps, data_path=files_dir_path)

    video_path = args.video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: video not opened correctly")

    lost_frames = 0
    pos_frames = 0
    skip_frames = args.skip_frames

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            output, rois, painting_bbs = get_bb(frame, include_steps=args.include_steps)
            people_bbs = yolo.get_people_bb(frame, painting_bbs)

            # TODO: Cristian should refactor this into a function
            for id, person_bb in enumerate(people_bbs):
                x, y, w, h = person_bb
                if args.include_steps:
                    fy, fx, _ = frame.shape
                    x = x // 2 + fx // 2
                    y = y // 2 + fy // 2
                    w //= 2
                    h //= 2
                if np.any(person_bb):
                    draw_bb(output, tl=(x, y), br=(x + w, y + h), color=(0, 0, 255), label="person_{}".format(id))

            cv2.imshow("Painting and people detection", output)

            key = cv2.waitKey(1)
            if key == ord('q'):  # quit
                break
            if key == ord('p'):  # pause
                cv2.waitKey(-1)
            if key == ord('r'):
                # retrieve paintings
                retrievals = []
                for i, roi in enumerate(rois):
                    rank, _ = retrieval.predict(roi, use_extra_check=True)
                    cv2.putText(roi, "{}".format(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, False)
                    print("Roi {} - rank = {}".format(i, rank))

                    # if retrieval fails
                    if rank[0] == -1:
                        print("Cannot retrieve painting for Roi {}".format(i))
                        retrievals.append(None)
                        out = show_on_row(roi, rectify(roi))
                        cv2.imshow("Roi {}".format(i), out)
                        continue

                    retrievals.append(rank[0])

                    ground_truth = cv2.imread(os.path.join(db_dir_path, "{:03d}.png".format(rank[0])))
                    warped, matches = sift_feature_matching_and_homography(roi, ground_truth,
                                                                           include_steps=args.include_steps)

                    # show output
                    if args.include_steps and warped is not None:
                        out = show_on_row(matches, warped)
                    elif warped is not None:
                        out = show_on_row(show_on_row(roi, ground_truth), warped)
                    else:
                        out = show_on_row(roi, ground_truth)
                    cv2.imshow("Roi {}".format(i), out)

                # localization
                for id, person_bb in enumerate(people_bbs):
                    room = people_locator.localize_person(person_bb, painting_bbs, retrievals, id=id, show_map=True)

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


if __name__ == '__main__':
    main()
