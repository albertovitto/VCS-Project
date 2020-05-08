import numpy as np
import cv2
import random
from utils import stack_frames, extract_rotated_rectangle

# https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
# https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=morphologyex#void%20morphologyEx(InputArray%20src,%20OutputArray%20dst,%20int%20op,%20InputArray%20kernel,%20Point%20anchor,%20int%20iterations,%20int%20borderType,%20const%20Scalar&%20borderValue)
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
# https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
# https://www.it-swarm.dev/it/python/ritaglia-rettangolo-restituito-da-minarearect-opencv-python/824441051/
# https://stackoverflow.com/questions/29739411/what-does-cv2-cv-boxpointsrect-return/51952289
# https://stackoverflow.com/questions/52782359/is-there-a-way-to-use-cv2-approxpolydp-to-approximate-open-curve


def painting_detection(frame):

    h, w, c = frame.shape
    original = frame.copy()

    gray_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray_bw, cv2.COLOR_GRAY2BGR)

    denoised = cv2.fastNlMeansDenoising(
        gray_bw, h=70, templateWindowSize=7, searchWindowSize=5)

    adap_th = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5
    )

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(adap_th, cv2.MORPH_CLOSE, kernel,
                               iterations=8, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    morph_grad = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel,
                                  iterations=5, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    _, contours, hierarchy = cv2.findContours(
        morph_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray_bw_cnt = gray_bw.copy()
    boxes = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 20000:
            cv2.rectangle(gray_bw_cnt, (x, y), (x+w, y+h), (0, 255, 0), 10)
            boxes.append((x, y, w, h))
            # cv2.drawContours(gray_bw_cnt, [box], -1, (0, 255, 0), 10)

    stacked_frames = stack_frames(
        gray_bw, denoised, adap_th, closing, morph_grad, gray_bw_cnt)
    return stacked_frames, boxes


def painting_detection_refine(frame, I):

    h, w = frame.shape
    original = frame.copy()

    adap_th = cv2.adaptiveThreshold(
        frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5
    )

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(adap_th, cv2.MORPH_CLOSE, kernel,
                               iterations=4, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    _, contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray_bw_cnt = original.copy()
    boxes = np.full(shape=(200, 200), fill_value=-1, dtype=np.int32)
    cropped_rectangle = []
    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if len(box) == 4 and cv2.isContourConvex(box) and cv2.contourArea(box) > 20000:
            # cv2.rectangle(gray_bw_cnt, (x, y), (x+w, y+h), (0, 255, 0), 10)
            cropped_rectangle = extract_rotated_rectangle(frame, I, box, rect)
            boxes = box
            cv2.drawContours(gray_bw_cnt, [box], -1, (0, 255, 0), 10)
            break

    stacked_frames = stack_frames(
        frame, adap_th, closing, gray_bw_cnt)

    return stacked_frames, cropped_rectangle, boxes
