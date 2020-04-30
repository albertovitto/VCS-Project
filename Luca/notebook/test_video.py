import numpy as np
import cv2
from vcsp.utils import horizontal_stack, vertical_stack, auto_alpha_beta


def find_paintings(img):
    alpha, beta = auto_alpha_beta(img)
    high_contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    gray_img = cv2.cvtColor(high_contrast_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5)

    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((5, 5)), iterations=1)

    edges = cv2.Canny(morph, 30, 60)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        maxc = cv2.contourArea(max(contours, key=cv2.contourArea))

        for i, c in enumerate(contours):
            epsilon = (len(c) / (3 * 4)) * 2  # 4 == number of desired points
            contours_poly = cv2.approxPolyDP(c, epsilon, True)
            x, y, w, h = cv2.boundingRect(contours_poly)

            if cv2.contourArea(c) > (maxc - maxc * 0.2):
                if cv2.isContourConvex(contours_poly):
                    cv2.drawContours(img, [c], 0, (0, 0, 255), 2)
                    cv2.drawContours(img, [contours_poly], 0, (255, 0, 0), 2)

    hstack1 = horizontal_stack(blur, th)
    hstack2 = horizontal_stack(morph, img)
    vstack = vertical_stack(hstack1, hstack2)
    cv2.imshow("Frame", vstack)


def main():
    video_name = "../../GOPR5826.MP4"
    video = cv2.VideoCapture(video_name)

    if not video.isOpened():
        print("Error")

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            find_paintings(frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()