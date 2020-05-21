# VCS-Project
This project was developed as part of the Vision and Cognitive Systems course as an exam in the Master's Degree in Computer Engineering in Unimore.

## Contributors
- Luca Denti
- Cristian Mercadante
- Alberto Vitto

## Tasks
Given a dataset of videos taken in _"Gallerie Estensi"_ in Modena together with pictures of its paintings, it was required to implement a software in Python capable of detecting paintings in videos and retrieve the original image from the dataset.

In particular:
- **Painting detection**
    - [x] Given an input video, your code should output a list of bounding boxes `(x, y, w, h)`, being `(x, y)` the upper left corner, each containing one painting.
    - [x] Create an interface to visualize given an image the ROI of a painting.
    - [x] Select painting and discard other artifacts.
    - [ ] _Optional_: segment precisely paintings with frames and also statues.
- **Painting rectification**
    - [x] Given an input video and detections (from the previous point), the code should output a new image for each painting, containing the rectified version of the painting.
    - [x] Pay attention to not-squared paintings.
- **Painting retrieval**
    - [x] Given one rectified painting (from the previous point), the code should return a ranked list of all the images in the painting DB, sorted by descending similarity with the detected painting. Ideally, the first retrieved item should be the picture of the detected painting.
- **People detection**
    - [x] Given an input video, the code should output a list of bounding boxes `(x, y, w, h)`, being `(x, y)` the upper left corner, each containing one person.
- **People localization**
    - [ ] Given an input video and people bounding boxes (from the previous point), the code should assign each person to one of the rooms of the Gallery. To do that, exploit the painting retrieval procedure (third point), and the mapping between paintings and rooms (in `data.csv`). Also, a map of the Gallery is available (`map.png`) for a better visualization.
    
Optional tasks:
- [ ] Given an input video, people and paintings' detections, determine whether each person is facing a painting or not.
- [ ] Given a view taken from the 3D model, detect each painting and replace it with its corresponding picture in the paintings DB, appropriately deformed to match the 3D view.
- [ ] Determine the distance of a person to the closest door: find the door, find the walls and the floor, try to compensate and predict distance.