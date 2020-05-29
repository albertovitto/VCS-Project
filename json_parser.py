import json


def json_parser(path):
    with open(path, 'r') as f:
        parser = json.load(f)

    imagePath = parser["imagePath"]
    imageHeight = parser["imageHeight"]
    imageWidth = parser["imageWidth"]
    print("Image details:  path: {}, H: {}, W: {}".format(
        imagePath, imageHeight, imageWidth))

    for painting in parser["shapes"]:
        if painting["label"] != "-1" and painting["label"] != "-2" and painting["label"] != "-3":
            ID = painting["label"]
            x1, y1 = painting["points"][0]
            x2, y2 = painting["points"][1]

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 < 0:
                x2 = 0
            if y2 < 0:
                y2 = 0

            tlx = 0
            tly = 0
            brx = 0
            bry = 0

            if x1 <= x2 and y1 >= y2:
                tlx = x1
                tly = y1
                brx = x2
                bry = y2

            if x2 <= x1 and y2 >= y1:
                tlx = x2
                tly = y2
                brx = x1
                bry = y1

            if x1 <= x2 and y1 <= y2:
                tlx = x1
                tly = y2
                brx = x2
                bry = y1

            if x2 <= x1 and y2 <= y1:
                tlx = x2
                tly = y1
                brx = x1
                bry = y2
            print("Painting with ID {} has coordinates tl_x: {}, tl_y: {}, br_x: {}, br_y: {}".format(
                ID, tlx, tly, brx, bry))


if __name__ == '__main__':
    path = '/home/alberto/PycharmProjects/VCS-Project/dataset/painting_retrival_test_set_alberto fino a 003_13_labelme/001_0.json'
    json_parser(path)
