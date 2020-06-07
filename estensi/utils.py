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