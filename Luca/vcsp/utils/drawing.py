import cv2


def draw_bb(img, tl, br, color, label=None):
    cv2.rectangle(img, tl, br, color, 2)

    if label is not None:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        br_label = tl[0] + label_size[0] + 3, tl[1] + label_size[1] + 4
        cv2.rectangle(img, tl, br_label, color, -1)
        cv2.putText(img, label, (tl[0], tl[1] + label_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    return img