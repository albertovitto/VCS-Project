import numpy as np
import cv2

"""
alpha:
    112x+36y+z=1.5
    118x+44y+z=2
    95x+48y+z=1.5 

beta:
    118x+44y+z=50
    112x+36y+z=20
    90x+36y+z=30
"""

def auto_alpha_beta(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    brightness = hsv_planes[2]
    mean, std = cv2.meanStdDev(brightness)

    acoeff1 = -3 / 340
    #acoeff1 = 3 / 104
    acoeff2 = 47 / 680
    #acoeff2 = 17 / 416
    acoeff3 = 0
    #acoeff3 = - 333 / 104

    alpha = acoeff1 * mean + acoeff2 * std + acoeff3
    if alpha >= 2.5:
        alpha = 2.5
    if alpha < 1:
        alpha = 1

    #bcoeff1 = -23 / 17
    bcoeff1 = -5 / 11
    #bcoeff2 = 81 / 17
    bcoeff2 = 45 / 11
    bcoeff3 = - 840 / 11

    beta = bcoeff1 * mean + bcoeff2 * std + bcoeff3
    if beta > 50:
        beta = 50

    beta = std - 10
    if beta < 0:
        beta = 0
    alpha = 1.5
    #print("mean = {}, std = {}, {} {}".format(mean, std, alpha, beta))
    return alpha, beta


def apply_brightness_contrast(input_img, brightness=255, contrast=127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
