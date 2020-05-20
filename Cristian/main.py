import cv2
from Cristian.image_processing.detection import get_bb
from Cristian.image_processing.retrieval_utils import sift_feature_matching_and_homography, retrieve, \
    get_painting_info_from_csv
from Luca.vcsp.painting_rectification.rectification import rectify

if __name__ == '__main__':

    lost_frames = 0
    pos_frames = 0
    skip_frames = True

    # video_name = '000/VIRB0393.MP4'
    video_name = '001/GOPR5826.MP4'
    # video_name = '011/3.mp4'
    # video_name = '003/GOPR1940.MP4'
    # video_name = '002/20180206_114720.mp4'

    video_path = '../dataset/videos/%s' % video_name

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: video not opened correctly")
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            output, rois = get_bb(frame, include_steps=True)
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", 1280, 720)
            cv2.imshow("Frame", output)
            key = cv2.waitKey(1)
            if key == ord('q'):  # quit
                break
            if key == ord('p'):  # pause
                cv2.waitKey(-1)
            if key == ord('r'):  # show rois
                for i, roi in enumerate(rois):
                    # rect_roi = rectify(roi)
                    # cv2.imshow("Rectified roi {}".format(i), rect_roi)
                    _, _, _, number = retrieve(roi)
                    retrieved_img_filename = "{:03d}.png".format(number)
                    retrieved_img = cv2.imread("../dataset/paintings_db/{}".format(retrieved_img_filename))
                    warped, out = sift_feature_matching_and_homography(roi, retrieved_img)
                    if warped is not None:
                        title, author, room = get_painting_info_from_csv(number)
                        if title is not None:
                            h, w, _ = warped.shape
                            cv2.putText(warped, title, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                                        1, cv2.LINE_AA)

                            cv2.imshow("Warped ROI-{}".format(i), warped)
                            cv2.imshow("SIFT matching: ROI-{}".format(i), out)
                cv2.waitKey(-1)
                for i in range(len(rois)):
                    cv2.destroyWindow("Warped ROI-{}".format(i))
                    cv2.destroyWindow("SIFT matching: ROI-{}".format(i))
                    # cv2.destroyWindow("Rectified roi {}".format(i))

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
