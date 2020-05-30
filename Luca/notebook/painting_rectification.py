import cv2
from Luca.vcsp.painting_detection.detection import get_bb
from Luca.vcsp.painting_rectification.rectification import rectify, rectify_with_retrieval
from Luca.vcsp.painting_retrieval.retrieval import PaintingRetrieval

if __name__ == '__main__':

    use_retrieval = True
    include_matches = True

    if use_retrieval:
        db_dir_path = '../../dataset/paintings_db/'
        files_dir_path = '../../dataset/'

        retrieval = PaintingRetrieval(db_dir_path, files_dir_path)
        retrieval.train()

    # video_name = '000/VIRB0393.MP4'
    #video_name = '001/GOPR5826.MP4'
    # video_name = '005/GOPR2045.MP4'
    # video_name = '012/IMG_4086.MOV'
    # video_name = '005/GOPR2051.MP4'
    # video_name = '004/IMG_3803.MOV'
    # video_name = '008/VIRB0419.MP4'
    #video_name = '008/VIRB0427.MP4'
    # video_name = '012/IMG_4080.MOV'
    video_name = '002/20180206_114720.mp4'

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
            output, rois, _ = get_bb(frame, include_steps=True)
            cv2.imshow("Painting detection", output)

            key = cv2.waitKey(1)
            if key == ord('q'):  # quit
                break
            if key == ord('p'):  # pause
                cv2.waitKey(-1)
            if key == ord('r'):  # show rectified rois
                for i, roi in enumerate(rois):

                    if use_retrieval:
                        rank, _ = retrieval.predict(roi)
                        ground_truth = cv2.imread('../../dataset/paintings_db/' + "{:03d}.png".format(rank[0]))
                        rect_roi, matches = rectify_with_retrieval(roi, ground_truth)
                    else:
                        rect_roi = rectify(roi, include_steps=False)

                    if rect_roi is None:
                        cv2.imshow("Roi {}".format(i), matches)

                    if rect_roi is not None:
                        cv2.imshow("Rectified roi {}".format(i), rect_roi)
                        if use_retrieval and include_matches:
                            cv2.imshow("Matches roi {}".format(i), matches)

                cv2.waitKey(-1)
                for i, roi in enumerate(rois):
                    cv2.destroyWindow("Rectified roi {}".format(i))
                    if use_retrieval and include_matches:
                        cv2.destroyWindow("Matches roi {}".format(i))

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