
import numpy as np
import cv2
from scipy.spatial import distance


def find_room_for_person(people_boxes, painting_boxes, painting_rooms):

    assert len(painting_boxes) == len(painting_rooms)
    # for each painting box I must have his room
    if not people_boxes:
        return None

    people_locations = np.full(shape=len(people_boxes), fill_value=-1)
    for i, (x1, y1, w1, h1) in enumerate(people_boxes):

        min_distance = +np.inf
        for j, (x2, y2, w2, h2) in enumerate(painting_boxes):
            X1 = x1+w1
            Y1 = y1+h1
            X2 = x2+w2
            Y2 = y2+h2

            left = X2 < x1
            right = X1 < x2
            bottom = Y2 < y1
            top = Y1 < y2

            if top and left:
                if distance.euclidean((x1, Y1), (X2, y2)) < min_distance:
                    min_distance = distance.euclidean((x1, Y1), (X2, y2))
                    people_locations[i] = painting_rooms[j]
            elif left and bottom:
                if distance.euclidean((x1, y1), (X2, Y2)) < min_distance:
                    min_distance = distance.euclidean((x1, y1), (X2, Y2))
                    people_locations[i] = painting_rooms[j]
            elif bottom and right:
                if distance.euclidean((X1, y1), (x2, Y2)) < min_distance:
                    min_distance = distance.euclidean((X1, y1), (x2, Y2))
                    people_locations[i] = painting_rooms[j]
            elif right and top:
                if distance.euclidean((X1, Y1), (x2, y2)) < min_distance:
                    min_distance = distance.euclidean((X1, Y1), (x2, y2))
                    people_locations[i] = painting_rooms[j]
            elif left:
                if x1 - X2 < min_distance:
                    min_distance = x1 - X2
                    people_locations[i] = painting_rooms[j]
            elif right:
                if x2 - X1 < min_distance:
                    min_distance = x2 - X1
                    people_locations[i] = painting_rooms[j]
            elif bottom:
                if y1 - Y2 < min_distance:
                    min_distance = y1 - Y2
                    people_locations[i] = painting_rooms[j]
            elif top:
                if y2 - Y1 < min_distance:
                    min_distance = y2 - Y1
                    people_locations[i] = painting_rooms[j]
            else:  # rectangles intersect
                min_distance = 0
                people_locations[i] = painting_rooms[j]

    return people_locations


people_boxes = [[1260, 474, 200, 400], [1430, 530, 100, 300]]
painting_boxes = [[369, 187, 300, 500], [739, 272, 200, 400],
                  [1040, 360, 100, 300], [1260, 460, 150, 200]]
# box -> retrival -> painting_id -> room
painting_rooms = [5, 5, 5, 6]
# goal: person on the roght should be assigned to room 6
person_room = find_room_for_person(
    people_boxes, painting_boxes, painting_rooms)
if person_room is not None:
    for i in range(len(people_boxes)):
        print("person #{} is in room {}".format(i, person_room[i]))
