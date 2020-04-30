import numpy as np
import cv2
from vcsp.utils import auto_alpha_beta
import matplotlib.pyplot as plt


def main():
    img = cv2.imread("../../img.png")
    alpha, beta = auto_alpha_beta(img)
    print(alpha)
    print(beta)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()