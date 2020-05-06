import cv2
from Cristian.image_processing.detection import get_bb
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
                    rect_roi = rectify(roi)
                    cv2.imshow("Rectified roi {}".format(i), rect_roi)
                cv2.waitKey(-1)
                for i, roi in enumerate(rois):
                    cv2.destroyWindow("Rectified roi {}".format(i))

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
