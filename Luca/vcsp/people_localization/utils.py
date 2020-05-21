import os
import copy

import numpy as np
import cv2

MAP_IMG_PATH = os.path.join("..", "..", "dataset", "map.png")


def get_map_rooms_coords():
    # room_num: (tl_x, tl_y, br_x, br_y)

    d = {
        1: (938, 408, 1031, 600),
        2: (938, 611, 1031, 700),
        3: (833, 611, 927, 700),
        4: (729, 611, 826, 700),
        5: (624, 611, 721, 700),
        6: (520, 611, 613, 700),
        7: (417, 611, 508, 700),
        8: (361, 611, 409, 700),
        9: (265, 611, 353, 700),
        10: (215, 611, 256, 700),
        11: (115, 611, 206, 700),
        12: (15, 611, 104, 700),
        13: (15, 409, 104, 600),
        14: (15, 310, 104, 399),
        15: (15, 115, 104, 298),
        16: (15, 17, 139, 104),
        17: (151, 17, 216, 104),
        18: (226, 17, 305, 104),
        19: (115, 115, 305, 399),
        20: (115, 409, 407, 600),
        21: (418, 409, 715, 600),
        22: (727, 409, 927, 600)
    }

    return d


def highlight_map_room(room_number):
    map_img = cv2.imread(MAP_IMG_PATH)
    overlay = copy.deepcopy(map_img)
    alpha = 0.5

    if room_number in get_map_rooms_coords().keys():
        tl_x, tl_y, br_x, br_y = get_map_rooms_coords()[room_number]
        cv2.rectangle(overlay, (tl_x, tl_y), (br_x, br_y), (79, 79, 255), -1)
        cv2.addWeighted(overlay, alpha, map_img, 1 - alpha, 0, map_img)

    return map_img
