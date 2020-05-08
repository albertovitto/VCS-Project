import numpy as np
import cv2
import random


def resize_image(scale_percent, image):
    # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def painting_detection(frame):

    h, w, c = frame.shape
    original = frame.copy()
    gray_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray_bw, cv2.COLOR_GRAY2BGR)

    denoised = cv2.fastNlMeansDenoising(
        gray_bw, h=7, templateWindowSize=7, searchWindowSize=3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_bw)

    adap_th = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5
    )

    adap_th = cv2.medianBlur(adap_th, 3)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(adap_th, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=4)

    _, contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray_bw_cnt = gray_bw.copy()

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 20000:
            cv2.rectangle(gray_bw_cnt, (x, y), (x+w, y+h), (0, 255, 0), 10)
            #cv2.drawContours(gray_bw_cnt, [approx], -1, (0, 255, 0), 10)
        # if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 10000:
        #     cv2.rectangle(gray_bw_cnt, (x, y), (x+w, y+h), (0, 255, 0), 10)
            #     cv2.drawContours(gray_bw_cnt, [approx], -1, (0, 255, 0), 10)

    a = np.hstack((gray_bw, equalized, denoised))
    b = np.hstack((erosion, dilation, gray_bw_cnt))
    output = np.concatenate((a, b), axis=0)
    return output


def main():
    # 000/VIRB0400.MP4  000/VIRB0406.MP4 720  002/20180206_114720.mp4
    # 005/GOPR2043.MP4 1080  011/3.mp4 2k
    #video_path = "dataset/videos/000/VIRB0400.MP4"
    #video_path = "dataset/videos/000/VIRB0406.MP4"
    video_path = "dataset/videos/002/20180206_114720.mp4"
    #video_path = "dataset/videos/003/GOPR1940.MP4"
    # video_path = "dataset/videos/005/GOPR2043.MP4"
    #video_path = "dataset/videos/003/GOPR1940.MP4"
    #video_path = "dataset/videos/011/3.mp4"
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Cant open video")

    lost_frames = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            roi = painting_detection(frame)
            if roi is not None:
                while roi.shape[1] > 1920:
                    roi = resize_image(50, roi)

                cv2.imshow("Painting detection results:", roi)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("p"):
                cv2.waitKey(-1)

        else:
            lost_frames += 1
            print("Lost frames: ", lost_frames)
            if lost_frames > 10:
                print("Too many errors reading video or video ended")
                break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
