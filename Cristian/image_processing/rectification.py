import cv2
import numpy as np
from Luca.vcsp.utils.multiple_show import horizontal_stack

from collections import defaultdict


# https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def get_mean_coordinate(intersections):
    coord = [tuple(i[0]) for i in intersections]
    coord = np.array(coord, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    centers = cv2.kmeans(coord, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[2]
    return centers


def main():
    img = cv2.imread("../roi/out2.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 5)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    canny = cv2.Canny(th, 5, 100, 5)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 100)

    frame = np.ndarray.copy(img)
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented)
    centers = get_mean_coordinate(intersections)
    for i in centers:
        point = (i[0], i[1])
        cv2.circle(frame, point, 5, (255, 0, 0), thickness=5)

    hstack = horizontal_stack(canny, frame)

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI", 1280, 720)
    cv2.imshow("ROI", hstack)
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
