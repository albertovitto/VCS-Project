import numpy as np
import cv2
import random
from utils import stack_frames


def painting_rectification(image):

    adap_th = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5
    )

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(adap_th, cv2.MORPH_CLOSE, kernel,
                               iterations=8, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    morph_grad = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel,
                                  iterations=5, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # Find contours in image
    _, contours, hierarchy = cv2.findContours(
        morph_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by contour area, as paintings will likely have larger contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    box = np.full(shape=(200, 200), fill_value=-1, dtype=np.int32)
    found = False
    # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html #4
    # Approximate the contours into a polygon (starting with the largest contour)
    # This will yield the first polygon with 4 points
    for contour in contours:

        epsilon = 0.05 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        if len(polygon) == 4 and cv2.isContourConvex(polygon) and cv2.contourArea(polygon) > 20000:
            box = polygon
            found = True
            break

    # TODO: handling additional polygons
    if not found:
        print('Could not find bounding box of 4 points')
        return [], box

    box = box.reshape(4, 2)
    approx_box = np.zeros((4, 2), dtype="float32")

    # Caclulate sum
    sum = box.sum(axis=1)
    approx_box[0] = box[np.argmin(sum)]
    approx_box[2] = box[np.argmax(sum)]

    # Calculate difference
    diff = np.diff(box, axis=1)
    approx_box[1] = box[np.argmin(diff)]
    approx_box[3] = box[np.argmax(diff)]

    # Determine width and height of bounding box
    smallest_x = 1000000
    smallest_y = 1000000
    largest_x = -1
    largest_y = -1

    for point in approx_box:
        if point[0] < smallest_x:
            smallest_x = point[0]
        if point[0] > largest_x:
            largest_x = point[0]
        if point[1] < smallest_y:
            smallest_y = point[1]
        if point[1] > largest_y:
            largest_y = point[1]

    maxWidth = int(largest_x - smallest_x)
    maxHeight = int(largest_y - smallest_y)

    bounding_box = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype="float32")

    # Apply transformation
    transform = cv2.getPerspectiveTransform(approx_box, bounding_box)
    result = cv2.warpPerspective(image, transform, (0, 0))

    # Crop out of original picture
    extracted = result[0:maxHeight, 0:maxWidth]

    show = stack_frames(image, adap_th, closing)
    return show, extracted


# def order_points(pts):
#     # initialzie a list of coordinates that will be ordered
#     # such that the first entry in the list is the top-left,
#     # the second entry is the top-right, the third is the
#     # bottom-right, and the fourth is the bottom-left
#     rect = np.zeros((4, 2), dtype="float32")
#     # the top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     # now, compute the difference between the points, the
#     # top-right point will have the smallest difference,
#     # whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#     # return the ordered coordinates
#     return rect


# def four_point_transform(image, pts):
#     # obtain a consistent order of the points and unpack them
#     # individually
#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect
#     # compute the width of the new image, which will be the
#     # maximum distance between bottom-right and bottom-left
#     # x-coordiates or the top-right and top-left x-coordinates
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))
#     # compute the height of the new image, which will be the
#     # maximum distance between the top-right and bottom-right
#     # y-coordinates or the top-left and bottom-left y-coordinates
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))
#     # now that we have the dimensions of the new image, construct
#     # the set of destination points to obtain a "birds eye view",
#     # (i.e. top-down view) of the image, again specifying points
#     # in the top-left, top-right, bottom-right, and bottom-left
#     # order
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")
#     # compute the perspective transform matrix and then apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#     # return the warped image
#     return warped


# def painting_rectification(frame):

#     box_points = find_surrounding_box(frame)
#     undistorted = four_point_transform(frame, box_points)
#     return undistorted


# def find_surrounding_box(frame):
#     adap_th = cv2.adaptiveThreshold(
#         frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5
#     )

#     kernel = np.ones((3, 3), np.uint8)
#     closing = cv2.morphologyEx(adap_th, cv2.MORPH_CLOSE, kernel,
#                                iterations=4, borderType=cv2.BORDER_CONSTANT, borderValue=0)

#     _, contours, hierarchy = cv2.findContours(
#         closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     external_box = None
#     for i, contour in enumerate(contours):
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         if len(box) == 4 and cv2.isContourConvex(box) and cv2.contourArea(box) > 20000:
#             external_box = box
#             break

#     return external_box

    # copy = image.copy()
    # kernel = np.ones((3, 3), np.uint8)
    # adap_th = cv2.adaptiveThreshold(
    #     image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5
    # )

    # closing = cv2.morphologyEx(adap_th, cv2.MORPH_OPEN, kernel,
    #                            iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # morph_grad = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel,
    #                               iterations=5, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # # Find contours in image
    # _, contours, hierarchy = cv2.findContours(
    #     morph_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Sort the contours by contour area, as paintings will likely have larger contours
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # external_box = None
    # # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html #4
    # # Approximate the contours into a polygon (starting with the largest contour)
    # # This will yield the first polygon with 4 points
    # for contour in contours:
    #     # TODO, goede waarde vinden
    #     # rect = cv2.minAreaRect(contour)
    #     # box = cv2.boxPoints(rect)
    #     # box = np.int0(box)
    #     epsilon = 0.05 * cv2.arcLength(contour, True)
    #     box = cv2.approxPolyDP(contour, epsilon, True)
    #     # box = cv2.boxPoints(rect)
    #     # box = np.int0(box)
    #     if len(box) == 4 and cv2.isContourConvex(box):
    #         if external_box == None:
    #             external_box = box
    #             break
    #         #cv2.drawContours(copy, [box], -1, (0, 255, 0), 10)

    # # TODO: handling additional polygons
    # if external_box is None:
    #     print('Could not find bounding box of 4 points')
    #     return None, None

    # external_box = external_box.reshape(4, 2)
    # approx_box = np.zeros((4, 2), dtype="float32")

    # # Caclulate sum
    # sum = external_box.sum(axis=1)
    # approx_box[0] = external_box[np.argmin(sum)]
    # approx_box[2] = external_box[np.argmax(sum)]

    # # Calculate difference
    # diff = np.diff(external_box, axis=1)
    # approx_box[1] = external_box[np.argmin(diff)]
    # approx_box[3] = external_box[np.argmax(diff)]

    # # Determine width and height of bounding box
    # smallest_x = 1000000
    # smallest_y = 1000000
    # largest_x = -1
    # largest_y = -1

    # for point in approx_box:
    #     if point[0] < smallest_x:
    #         smallest_x = point[0]
    #     if point[0] > largest_x:
    #         largest_x = point[0]
    #     if point[1] < smallest_y:
    #         smallest_y = point[1]
    #     if point[1] > largest_y:
    #         largest_y = point[1]

    # maxWidth = int(largest_x - smallest_x)
    # maxHeight = int(largest_y - smallest_y)

    # bounding_box = np.array([
    #     [0, 0],
    #     [maxWidth, 0],
    #     [maxWidth, maxHeight],
    #     [0, maxHeight]], dtype="float32")

    # # Apply transformation
    # transform = cv2.getPerspectiveTransform(approx_box, bounding_box)
    # result = cv2.warpPerspective(image, transform, (0, 0))

    # # Crop out of original picture
    # extracted = result[0:maxHeight, 0:maxWidth]

    # show = stack_frames(image, adap_th, closing, morph_grad, copy)
    # return show, extracted
