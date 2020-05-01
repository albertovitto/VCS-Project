import cv2
from Cristian.image_processing.detection import get_bb

if __name__ == '__main__':

    frame_skip = 0

    # video_name = '000/VIRB0393.MP4'
    video_name = '001/GOPR5826.MP4'
    # video_name = '011/3.mp4'
    # video_name = '003/GOPR1940.MP4'
    # video_name = '002/20180206_114720.mp4'
    video_path = '../dataset/videos/%s' % video_name
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: video not opened correctly")
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        i += 1
        if ret and (frame_skip == 0 or i % frame_skip == 0):
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", 1280, 720)
            cv2.imshow("Frame", get_bb(frame))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1)
        if ret is False:
            break

    video.release()
    cv2.destroyAllWindows()
