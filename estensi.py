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
from Luca.vcsp.painting_detection.detection import get_bb
from Luca.vcsp.painting_retrieval.retrieval import PaintingRetrieval
from Luca.vcsp.people_localization.utils import highlight_map_room
from Luca.vcsp.utils.drawing import draw_bb
from Luca.vcsp.utils.multiple_show import show_on_row


def arg_parse():
    parser = argparse.ArgumentParser(description='Vision and Cognitive Systems project: Gallerie Estensi')
    parser.add_argument('--video', dest='video', help='path of the selected video', type=str, required=True)
    parser.add_argument('--include_steps', dest='include_steps', action='store_true',
                        help='toggles additional screens that show more info (like intermediate steps)')
    return parser.parse_args()


def main():
    args = arg_parse()

    path = os.path.abspath(os.path.dirname(__file__))
    db_dir_path = os.path.join(path, "dataset", "paintings_db")
    files_dir_path = os.path.join(path, "dataset")

    # FIXME: cannot read features_db because of path
    #
    # WORKING FIX:
    ### Luca/vcsp/painting_retrieval/utils.py, line 209
    # imgs_path_list.append(os.path.join(db_dir_path, filename))
    #
    ### Luca/vcsp/painting_retrieval/retrieval.py, lines 15-16
    # self.features_db = np.load(os.path.join(files_dir_path, 'features_db.npy'))
    # self.img_features_db = np.load(os.path.join(files_dir_path, 'img_features_db.npy'))
    retrieval = PaintingRetrieval(db_dir_path, files_dir_path)
    retrieval.train()

    yolo = Yolo(yolo_path=os.path.join(path, "Cristian", "YOLOv3"))

    video_path = args.video

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: video not opened correctly")

    lost_frames = 0
    pos_frames = 0
    skip_frames = True

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            output, rois, painting_bbs = get_bb(frame, include_steps=args.include_steps)
            people_bbs = yolo.get_people_bb(frame, painting_bbs)

            # TODO: Cristian will include this piece of code into get_people_bb and return both output and people_bbs
            for person_bb in people_bbs:
                x, y, w, h = person_bb
                if np.any(person_bb):
                    draw_bb(output, tl=(x, y), br=(x + w, y + h), color=(0, 0, 255), label="person")

            cv2.imshow("Painting detection", output)

            key = cv2.waitKey(1)
            if key == ord('q'):  # quit
                break
            if key == ord('p'):  # pause
                cv2.waitKey(-1)
            if key == ord('r'):  # show rois with image retrieval
                retrievals = []
                for i, roi in enumerate(rois):
                    rank, _ = retrieval.predict(roi)
                    cv2.putText(roi, "{}".format(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, False)
                    print("Roi {} - rank = {}".format(i, rank))
                    ground_truth = cv2.imread(os.path.join(db_dir_path, "{:03d}.png".format(rank[0])))
                    cv2.imshow("Roi {}".format(i), show_on_row(roi, ground_truth))
                    retrievals.append(rank[0])
                for person_bb in people_bbs:
                    room = pl.localize_person(person_bb, painting_bbs, retrievals,
                                              distance=pl.CENTER_DISTANCE,
                                              weighting=pl.AREA,
                                              voting=True,
                                              verbose=args.include_steps,
                                              data_path=files_dir_path)

                    # FIXME: cannot read map because of path
                    # Possible fix: path as function argument
                    map_img = highlight_map_room(room)
                    cv2.imshow("People localization", map_img)

                # TODO: add rectification (use include_steps to show SIFT matches)

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