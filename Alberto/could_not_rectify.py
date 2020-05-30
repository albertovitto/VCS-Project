import numpy as np
import cv2


def could_not_rectify ():
    img = np.zeros((512, 512, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, 'COULD NOT RECTIFY',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    return img