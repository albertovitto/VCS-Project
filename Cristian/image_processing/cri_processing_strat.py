import cv2
import numpy as np

'''
NOTES:
## OTSU
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th = cv2.bitwise_not(th)

## CANNY
canny = cv2.Canny(gray, 50, 100, 5)
'''


def frame_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 75, 75)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 5)
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=3)
    morph = cv2.dilate(morph, np.ones((3, 3)))
    return blur, th, morph
