import json
import os

import cv2
import numpy as np


def draw_bb(img, tl, br, color, label=None):
    cv2.rectangle(img, tl, br, color, 2)

    if label is not None:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        br_label = tl[0] + label_size[0] + 3, tl[1] + label_size[1] + 4
        cv2.rectangle(img, tl, br_label, color, -1)
        cv2.putText(img, label, (tl[0], tl[1] + label_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    return img


def show_on_row(img1, img2):
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    height = max([img1.shape[0], img2.shape[0]])
    width = img1.shape[1] + img2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)

    output[0:img1.shape[0], 0:img1.shape[1], :] = img1
    output[0:img2.shape[0], img1.shape[1]:, :] = img2

    output = resize_to_fit(output)

    return output


def show_on_col(img1, img2):
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    height = img1.shape[0] + img2.shape[0]
    width = max([img1.shape[1], img2.shape[1]])

    output = np.zeros((height, width, 3), dtype=np.uint8)

    output[0:img1.shape[0], 0:img1.shape[1], :] = img1
    output[img1.shape[0]:, 0:img2.shape[1]:, :] = img2

    output = resize_to_fit(output)

    return output


def resize_to_fit(img, dw=1920, dh=1080):
    h, w = img.shape[0:2]
    dist_w = max(0, w - dw)
    dist_h = max(0, h - dh)

    if dist_h == 0 and dist_w == 0:
        return img

    if dist_h > dist_w:
        scale_factor = (1 - dist_h / h)
    else:
        scale_factor = (1 - dist_w / w)

    width = int(w * scale_factor)
    height = int(h * scale_factor)
    output = cv2.resize(img, (width, height))
    return output


def get_ground_truth_bbs(ground_truth_set_dir_path, video, frame):

    def fit_in_range(points, img_shape):
        for i in range(len(points)):
            if points[i][0] < 0:
                points[i][0] = 0
            elif points[i][0] > img_shape[1]:
                points[i][0] = img_shape[1]

            if points[i][1] < 0:
                points[i][1] = 0
            elif points[i][1] > img_shape[0]:
                points[i][1] = img_shape[0]

        return np.array(points).astype(int)

    def extract_bb_coords(points):
        p1 = points[0]
        p2 = points[1]

        tl = 0
        br = 0
        if p1[0] < p2[0]:  # p1 is on LEFT, p2 in on RIGHT
            if p1[1] < p2[1]:  # p1 is TOP-LEFT, p2 is BOTTOM-RIGHT
                tl = p1
                br = p2
            else:  # p1 is BOTTOM-LEFT, p2 is TOP-RIGHT
                tl = [p1[0], p2[1]]
                br = [p2[0], p1[1]]
        else:  # p1 is on RIGHT, p2 is on LEFT
            if p1[1] < p2[1]:  # p1 is TOP-RIGHT, p2 in BOTTOM-LEFT
                tl = [p2[0], p1[1]]
                br = [p1[0], p2[1]]
            else:  # p1 is BOTTOM-RIGHT, p2 is TOP-LEFT
                br = p1
                tl = p2

        w = br[0] - tl[0]
        h = br[1] - tl[1]

        return tl[0], tl[1], w, h

    # [0:94] = painting in db
    # -1 = painting not identified or not existing
    # -2 = person
    # -3 = statue
    bbs = {"paintings": [], "people": [], "statues": []}

    try:
        file = open(os.path.join(ground_truth_set_dir_path, "{}_{}.json".format(video, frame)), "r")
    except FileNotFoundError:
        return bbs

    data = json.load(file)
    img_h = data["imageHeight"]
    img_w = data["imageWidth"]
    for s in data["shapes"]:
        label = int(s["label"])
        points = fit_in_range(s["points"], img_shape=(img_h, img_w))
        bb = extract_bb_coords(points)
        if label >= 0 or label == -1:
            painting = {"bb": bb, "label": label}
            bbs["paintings"].append(painting)
        elif label == -2:
            bbs["people"].append(bb)
        elif label == -3:
            bbs["statues"].append(bb)
        else:
            continue

    file.close()

    return bbs


def bb_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = abs(max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)
