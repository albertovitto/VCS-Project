import numpy as np
import cv2
import math


def resize_image(scale_percent, image):
    # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# I want 3 frames for each row, independently of the number of the frames
def stack_frames(*argv):
    # each frames has NOT been resized before, they have all the same original size of the video
    if not argv:
        raise ValueError('Cant pass empty array to stack_frames function')

    frames = []

    for frame in argv:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frames.append(frame)

    h, w, c = frames[0].shape
    frames_seen = 0
    frames_tot = len(argv)
    ROWS = math.ceil(frames_tot/3)
    COLS = 3

    stacked_frames = np.zeros(shape=(ROWS*h, COLS*w, c), dtype=np.uint8)

    for row in range(ROWS):
        for col in range(COLS):
            #print(h*row, h*row+h, w*col, w*col+w)
            if frames_seen < frames_tot:
                stacked_frames[
                    h*row: h*row+h,
                    w * col: w*col+w,
                    :] = frames[frames_seen]
                frames_seen += 1

    while stacked_frames.shape[1] > 1920:
        stacked_frames = resize_image(scale_percent=90, stacked_frames)

    while stacked_frames.shape[0] > 960:
        stacked_frames = resize_image(scale_percent=90, stacked_frames)

    return stacked_frames
    # a = np.hstack((gray_bw, closing, morph_grad))
    # b = np.hstack((gray_bw_cnt, gray_bw_cnt, gray_bw_cnt))
    #output = np.concatenate((a, b), axis=0)


# def extract_rotated_rectangle(img, rect):

#     # rotate img
#     angle = rect[2]
#     rows, cols = img.shape[0], img.shape[1]
#     M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
#     img_rot = cv2.warpAffine(img, M, (cols, rows))

#     # rotate bounding box
#     rect0 = (rect[0], rect[1], 0.0)
#     box = cv2.boxPoints(rect0)
#     pts = np.int0(cv2.transform(np.array([box]), M))[0]
#     pts[pts < 0] = 0

#     # crop
#     img_crop = img_rot[pts[1][1]:pts[0][1],
#                        pts[1][0]:pts[2][0]]

#     return img_crop

def swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b


def extract_rotated_rectangle(frame, I, box, rect):
    # the order of the box points: bottom left, top left, top right,
    # bottom right

    # get width and height of the detected rectangle

    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = int(rect[2])
    print("#", I)
    print("angle ", angle)
    print("H W ", height, width)

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened

    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    # dst_pts = np.array([[0, 0],
    #                     [width-1, 0],
    #                     [0, height-1],
    #                     [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(frame, M, (width, height))

    if width > height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    print("\n")
    return warped


# def extract_rotated_rectangle(img, I, box, rect):
#     # rotate img
#     angle = rect[2]
#     rows, cols = img.shape[0], img.shape[1]
#     M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
#     img_rot = cv2.warpAffine(img, M, (cols, rows))

#     # rotate bounding box
#     rect0 = (rect[0], rect[1], 0.0)
#     box = cv2.boxPoints(rect0)
#     pts = np.int0(cv2.transform(np.array([box]), M))[0]
#     pts[pts < 0] = 0

#     # crop
#     img_crop = img_rot[pts[1][1]:pts[0][1],
#                        pts[1][0]:pts[2][0]]
#     return img_rot
