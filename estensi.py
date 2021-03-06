import argparse
import os
import cv2

from estensi.people_detection.detection import Yolo
import estensi.people_localization.localization as pl
from estensi.people_localization.localization import get_painting_info_from_csv
from estensi.painting_rectification.rectification import sift_feature_matching_and_homography
from estensi.painting_detection.detection import get_bb
from estensi.painting_rectification.rectification import rectify
from estensi.painting_retrieval.retrieval import PaintingRetrieval
from estensi.utils import draw_bb, show_on_row, resize_to_fit


def check_positive(value):
    try:
        ivalue = int(value)
    except:
        raise argparse.ArgumentTypeError("%s is not an integer" % value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def arg_parse():
    parser = argparse.ArgumentParser(description='Vision and Cognitive Systems project: Gallerie Estensi')
    parser.add_argument('--video', dest='video', help='path of the selected video', type=str, required=False)
    parser.add_argument('--folder', dest='folder', help='path of a folder with different videos', type=str)
    parser.add_argument('--include_steps', dest='include_steps', action='store_true',
                        help='toggles additional screens that show more info (like intermediate steps)')
    parser.add_argument('--skip_frames', dest='skip_frames', help='set amunt of frames to skip', type=check_positive,
                        required=False)
    return parser.parse_args()


def analyze_single_video(video_path, args, db_dir_path, files_dir_path, retrieval, yolo):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video {}...".format(video_path))

    lost_frames = 0
    pos_frames = 0
    skip_frames = args.skip_frames

    while video.isOpened():
        ret, frame = video.read()
        current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        if ret:
            output, rois, painting_bbs = get_bb(frame, include_steps=args.include_steps)
            people_bbs = yolo.get_people_bb(frame, painting_bbs)

            for id, person_bb in enumerate(people_bbs):
                x, y, w, h = person_bb
                if args.include_steps:
                    oy, ox, _ = output.shape
                    fy, fx, _ = frame.shape
                    scaling = oy / fy * 0.5
                    x = int(fx * scaling) + int(x * scaling)
                    y = int(fy * scaling) + int(y * scaling)
                    w = int(w * scaling)
                    h = int(h * scaling)

                draw_bb(output, tl=(x, y), br=(x + w, y + h), color=(0, 0, 255), label="person_{}".format(id))

            cv2.imshow("Painting and people detection", resize_to_fit(output, dw=1920, dh=900))
            key = cv2.waitKey(1)
            if key == ord('q'):  # quit
                break
            if key == ord('p'):  # pause
                print("+-----------------------------------------------+")
                print("Video paused at frame #{}".format(current_frame))
                print("+-----------------------------------------------+")
                cv2.waitKey(-1)
            if key == ord('r'):
                # retrieve paintings
                retrievals = []
                titles = []
                print("+-----------------------------------------------+")
                print("Video paused for retrieval at frame #{}".format(current_frame))

                for i, roi in enumerate(rois):
                    rank, _ = retrieval.predict(roi, use_extra_check=True)
                    print("Roi {} - rank = {}".format(i, rank))

                    retrieval_failed = rank[0] == -1
                    if retrieval_failed:
                        print("Cannot retrieve painting for Roi {}".format(i))
                        retrievals.append(None)
                        title = "Roi {} - Could not find matches and retrieve painting".format(i)
                        titles.append(title)

                        warped = rectify(roi)
                        matches = None
                    else:
                        title, author, room = get_painting_info_from_csv(painting_id=rank[0],
                                                                         path=os.path.join(files_dir_path, "data.csv"))
                        print("Title: {} \nAuthor: {} \nRoom: {}".format(title, author, room))
                        retrievals.append(rank[0])
                        title = title if str(author) == 'nan' else str(title) + " - " + str(author)
                        title = "Roi {} - {}".format(i, title)
                        titles.append(title)

                        ground_truth = cv2.imread(os.path.join(db_dir_path, "{:03d}.png".format(rank[0])))
                        warped, matches = sift_feature_matching_and_homography(roi, ground_truth,
                                                                               include_steps=args.include_steps)

                    # show output
                    if retrieval_failed:
                        if warped is not None:
                            out = show_on_row(roi, warped)
                        else:
                            out = roi
                    else:
                        if warped is not None:
                            if args.include_steps:
                                out = show_on_row(matches, warped)
                            else:
                                out = show_on_row(show_on_row(roi, ground_truth), warped)
                        else:
                            out = show_on_row(roi, ground_truth)

                    out = resize_to_fit(out, dw=1920, dh=900)
                    cv2.imshow("{}".format(title), out)

                # localization
                room, votes, map_img = pl.localize_paintings(retrievals, data_path=files_dir_path)
                if room is not None:
                    cv2.imshow("Room", resize_to_fit(map_img))
                    if args.include_steps:
                        print("Room votes: {}".format(votes))
                else:
                    cv2.imshow("Cannot find room", resize_to_fit(map_img))

                print("+-----------------------------------------------+")
                # destroy windows
                cv2.waitKey(-1)
                for title in titles:
                    cv2.destroyWindow(title)

                if room is not None:
                    cv2.destroyWindow("Room")
                else:
                    cv2.destroyWindow("Cannot find room")

            if skip_frames:
                pos_frames += skip_frames
                if pos_frames > video.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
                video.set(cv2.CAP_PROP_POS_FRAMES, pos_frames)
        else:
            lost_frames += 1
            if lost_frames > 10:
                print("Too many errors reading video or video ended")
                break

    video.release()
    cv2.destroyAllWindows()


def main():
    args = arg_parse()

    path = os.path.abspath(os.path.dirname(__file__))
    db_dir_path = os.path.join(path, "dataset", "paintings_db")
    files_dir_path = os.path.join(path, "dataset")

    retrieval = PaintingRetrieval(db_dir_path, files_dir_path)
    retrieval.train()

    yolo = Yolo(yolo_path=os.path.join(path, "estensi", "people_detection"))

    if args.folder is not None and os.path.isdir(args.folder) is True:
        print("Folder to analyze: " + args.folder)
        for video_name in os.listdir(args.folder):
            video_path = os.path.join(args.folder, video_name)
            print("Analysing {}...".format(video_path))
            analyze_single_video(video_path, args, db_dir_path, files_dir_path, retrieval, yolo)

    if args.video is not None:
        print("Video to analyze: " + args.video)
        print("Analysing {}...".format(args.video))
        analyze_single_video(args.video, args, db_dir_path, files_dir_path, retrieval, yolo)

    if args.folder is None and args.video is None:
        print("Please specify a valid folder or video path")


if __name__ == '__main__':
    main()
