import json
import os


path = os.path.join("..", "..", "..", "dataset", "ground_truth_alberto fino a 003_13_labelme_1")
print(len(os.listdir(path)))

for filename in os.listdir(path):
    with open(os.path.join(path, filename), "r") as jsonFile:
        data = json.load(jsonFile)

    image_path = data["imagePath"]
    image_path = "..\\painting_detection_test_set\\" + image_path
    data["imagePath"] = image_path

    with open(os.path.join(path, filename), "w") as jsonFile:
        json.dump(data, jsonFile)
