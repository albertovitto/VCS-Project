import argparse
import os
import cv2
import numpy as np
from estensi.people_detection.detection import Yolo
import estensi.people_localization.localization as pl
from estensi.people_localization.localization import PeopleLocator, get_painting_info_from_csv
from estensi.painting_rectification.rectification import sift_feature_matching_and_homography
from estensi.painting_detection.detection import get_bb
from estensi.painting_rectification.rectification import rectify
from estensi.painting_retrieval.retrieval import PaintingRetrieval
from estensi.utils import draw_bb, show_on_row, resize_image, could_not_find_matches, resize_to_fit


def arg_parse():
    parser = argparse.ArgumentParser(description='Vision and Cognitive Systems project: Gallerie Estensi')
    parser.add_argument('--video', dest='video', help='path of the selected video', type=str, required=False)
    parser.add_argument('--folder', dest='folder', help='path of a folder with different videos', type=str)
    parser.add_argument('--include_steps', dest='include_steps', action='store_true',
                        help='toggles additional screens that show more info (like intermediate steps)')
    parser.add_argument('--skip_frames', dest='skip_frames', action='store_true', help='toggles frame skip')
    return parser.parse_args()

def analyze_single_video(video_path,include_steps, skip_frames):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: video not opened correctly")

    lost_frames = 0
    pos_frames = 0
    skip_frames = skip_frames
    current_frame = 0

    while video.isOpened():
        ret, frame = video.read()
        current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        if ret:
            output, rois, painting_bbs = get_bb(frame, include_steps=include_steps)
            people_bbs = yolo.get_people_bb(frame, painting_bbs)

            # TODO: Cristian should refactor this into a function
            for id, person_bb in enumerate(people_bbs):
                x, y, w, h = person_bb
                if include_steps:
                    fy, fx, _ = frame.shape
                    x = x // 2 + fx // 2
                    y = y // 2 + fy // 2
                    w //= 2
                    h //= 2

                draw_bb(output, tl=(x, y), br=(x + w, y + h), color=(0, 0, 255), label="person_{}".format(id))

            cv2.imshow("Painting and people detection", resize_to_fit(output))
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
                    cv2.putText(roi, "{}".format(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, False)
                    print("Roi {} - rank = {}".format(i, rank))

                    # if retrieval fails
                    if rank[0] == -1:
                        print("Cannot retrieve painting for Roi {}".format(i))
                        retrievals.append(None)
                        titles.append(None)
                        out = show_on_row(roi, rectify(roi))
                        h, w, c = out.shape
                        out = np.hstack((out, could_not_find_matches(h, w, c)))
                        cv2.imshow("Roi {}".format(i), resize_to_fit(out))
                        continue

                    title, author, room = get_painting_info_from_csv(painting_id=rank[0],
                                                                     path=os.path.join(files_dir_path, "data.csv"))
                    print("Title: {} \nAuthor: {} \nRoom: {}".format(title, author, room))
                    retrievals.append(rank[0])
                    title = title if str(author) == 'nan' else str(title) + " - " + str(author)
                    titles.append(str(title))

                    ground_truth = cv2.imread(os.path.join(db_dir_path, "{:03d}.png".format(rank[0])))
                    warped, matches = sift_feature_matching_and_homography(roi, ground_truth,
                                                                           include_steps=include_steps)

                    # show output
                    if include_steps and warped is not None:
                        out = show_on_row(matches, warped)
                    elif warped is not None:
                        out = show_on_row(show_on_row(roi, ground_truth), warped)
                    else:
                        out = show_on_row(roi, ground_truth)
                        h, w, c = out.shape
                        out = np.hstack((out, could_not_find_matches(h, w, c)))
                    out = resize_to_fit(out, dw=1920, dh=1080)
                    cv2.imshow("{}".format(title), out)

                # localization
                # for id, person_bb in enumerate(people_bbs):
                #     room = people_locator.localize_person(person_bb, painting_bbs, retrievals, id=id, show_map=True)
                room = pl.localize_paintings(retrievals, data_path=files_dir_path, verbose=include_steps)

                print("+-----------------------------------------------+")
                cv2.waitKey(-1)
                for i, (title) in enumerate(titles):
                    if title is None:
                        cv2.destroyWindow("Roi {}".format(i))
                    else:
                        cv2.destroyWindow("{}".format(title))

                cv2.destroyWindow("Cannot find room")
                cv2.destroyWindow("Room")

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

def main():
    args = arg_parse()

    path = os.path.abspath(os.path.dirname(__file__))
    db_dir_path = os.path.join(path, "dataset", "paintings_db")
    files_dir_path = os.path.join(path, "dataset")

    retrieval = PaintingRetrieval(db_dir_path, files_dir_path)
    retrieval.train()

    yolo = Yolo(yolo_path=os.path.join(path, "estensi", "people_detection"))

    people_locator = PeopleLocator(distance=pl.CENTER_DISTANCE, weighting=pl.SQRT_AREA, voting=True,
                                   verbose=args.include_steps, data_path=files_dir_path)

    if args.folder is not None and os.path.isdir(args.folder) is True:
        folder_path = args.folder
        print("Folder to analyze: " + folder_path)
        for video_name in os.listdir(folder_path):
            print("Analysing {}...".format(folder_path + video_name))
            video_path = os.path.join(folder_path, video_name)
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print("Error: video {} not opened correctly".format(video_path))

            lost_frames = 0
            pos_frames = 0
            skip_frames = args.skip_frames
            current_frame = 0

            while video.isOpened():
                ret, frame = video.read()
                current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))

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

                        draw_bb(output, tl=(x, y), br=(x + w, y + h), color=(0, 0, 255), label="person_{}".format(id))

                    cv2.imshow("Painting and people detection", resize_to_fit(output))
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

                            # if retrieval fails
                            if rank[0] == -1:
                                print("Cannot retrieve painting for Roi {}".format(i))
                                retrievals.append(None)
                                titles.append(None)
                                out = show_on_row(roi, rectify(roi))
                                h, w, c = out.shape
                                out = np.hstack((out, could_not_find_matches(h, w, c)))
                                cv2.imshow("Roi {}".format(i), resize_to_fit(out))
                                continue

                            title, author, room = get_painting_info_from_csv(painting_id=rank[0],
                                                                             path=os.path.join(files_dir_path,
                                                                                               "data.csv"))
                            print("Title: {} \nAuthor: {} \nRoom: {}".format(title, author, room))
                            retrievals.append(rank[0])
                            title = title if str(author) == 'nan' else str(title) + " - " + str(author)
                            title = "Roi {} - {}".format(i, title)
                            titles.append(title)

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
                                h, w, c = out.shape
                                out = np.hstack((out, could_not_find_matches(h, w, c)))
                            out = resize_to_fit(out, dw=1920, dh=1080)
                            cv2.imshow("{}".format(title), out)

                        # localization
                        # for id, person_bb in enumerate(people_bbs):
                        #     room = people_locator.localize_person(person_bb, painting_bbs, retrievals, id=id, show_map=True)
                        room = pl.localize_paintings(retrievals, data_path=files_dir_path, verbose=args.include_steps)

                        print("+-----------------------------------------------+")
                        cv2.waitKey(-1)
                        for i, (title) in enumerate(titles):
                            if title is None:
                                cv2.destroyWindow("Roi {}".format(i))
                            else:
                                cv2.destroyWindow("{}".format(title))

                        cv2.destroyWindow("Cannot find room")
                        cv2.destroyWindow("Room")

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
            #analyze_single_video(folder_path + video, yolo, retrieval, include_steps, args.skip_frames)

    if args.video is not None:
        video_path = args.video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print("Error opening video {}...".format(video_path))

        lost_frames = 0
        pos_frames = 0
        skip_frames = args.skip_frames
        current_frame = 0

        while video.isOpened():
            ret, frame = video.read()
            current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))

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

                    draw_bb(output, tl=(x, y), br=(x + w, y + h), color=(0, 0, 255), label="person_{}".format(id))

                cv2.imshow("Painting and people detection", resize_to_fit(output))
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

                        # if retrieval fails
                        if rank[0] == -1:
                            print("Cannot retrieve painting for Roi {}".format(i))
                            retrievals.append(None)
                            titles.append(None)
                            out = show_on_row(roi, rectify(roi))
                            h, w, c = out.shape
                            out = np.hstack((out, could_not_find_matches(h, w, c)))
                            cv2.imshow("Roi {}".format(i), resize_to_fit(out))
                            continue

                        title, author, room = get_painting_info_from_csv(painting_id=rank[0], path=os.path.join(files_dir_path, "data.csv"))
                        print("Title: {} \nAuthor: {} \nRoom: {}".format(title, author, room))
                        retrievals.append(rank[0])
                        title = title if str(author) == 'nan' else str(title) + " - " + str(author)
                        title = "Roi {} - {}".format(i, title)
                        titles.append(title)

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
                            h, w, c = out.shape
                            out = np.hstack((out, could_not_find_matches(h, w, c)))
                        out = resize_to_fit(out, dw=1920, dh=1080)
                        cv2.imshow("{}".format(title), out)

                    # localization
                    # for id, person_bb in enumerate(people_bbs):
                    #     room = people_locator.localize_person(person_bb, painting_bbs, retrievals, id=id, show_map=True)
                    room = pl.localize_paintings(retrievals, data_path=files_dir_path, verbose=args.include_steps)

                    print("+-----------------------------------------------+")
                    cv2.waitKey(-1)
                    for i, (title) in enumerate(titles):
                        if title is None:
                            cv2.destroyWindow("Roi {}".format(i))
                        else:
                            cv2.destroyWindow("{}".format(title))

                    cv2.destroyWindow("Cannot find room")
                    cv2.destroyWindow("Room")

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

    if args.folder is None and args.video is None:
        print("Please specify a valid folder or video path")

if __name__ == '__main__':
    main()
