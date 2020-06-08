import numpy as np
import cv2


def could_not_find_matches(h, w, c):
    img = np.zeros((h, w, c), np.uint8)

    text = ['COULD', 'NOT', 'FIND', 'MATCHES', 'AND', 'RECTIFY']

    offset = int(h/len(text))
    x, y = 20, 20
    for idx, word in enumerate(text):
        cv2.putText(img, str(word), (x, y + offset * idx), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img