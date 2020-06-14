# VCS-Project
This project was developed as part of the Vision and Cognitive Systems course as an exam in the Master's Degree in Computer Engineering in Unimore.

## Contributors

| Name                | Email                                                           |
| ------------------- | --------------------------------------------------------------- |
| Luca Denti          | [`211805@studenti.unimore.it`](mailto:211805@studenti.unimore.it) |
| Cristian Mercadante | [`213808@studenti.unimore.it`](mailto:213808@studenti.unimore.it) |
| Alberto Vitto       | [`214372@studenti.unimore.it`](mailto:214372@studenti.unimore.it) |


## Tasks
Given a dataset of videos taken in _"Gallerie Estensi"_ in Modena together with pictures of its paintings, it was required to implement a software in Python capable of detecting paintings in videos and retrieve the original image from the dataset.

In particular:
- **Painting detection**
    - [x] Given an input video, the code should output a list of bounding boxes `(x, y, w, h)`, being `(x, y)` the upper left corner, each containing one painting.
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
    - [x] Given an input video and people bounding boxes (from the previous point), the code should assign each person to one of the rooms of the Gallery. To do that, exploit the painting retrieval procedure (third point), and the mapping between paintings and rooms (in `data.csv`). Also, a map of the Gallery is available (`map.png`) for a better visualization.
    
Optional tasks:
- [ ] Given an input video, people and paintings' detections, determine whether each person is facing a painting or not.
- [ ] Given a view taken from the 3D model, detect each painting and replace it with its corresponding picture in the paintings DB, appropriately deformed to match the 3D view.
- [ ] Determine the distance of a person to the closest door: find the door, find the walls and the floor, try to compensate and predict distance.

## Project structure
  ```bash
.
├── dataset
│   ├── data.csv
│   ├── features_db.npy
│   ├── ground_truth
│   │   ├── 000_0.json
│   │   ├── ...
│   │   └── 014_9.json
│   ├── img_features_db.npy
│   ├── map.png
│   ├── paintings_db
│   │   ├── 000.png
│   │   ├── ...
│   │   └── 094.png
│   ├── screenshots_3d_model
│   │   ├── 100916250_542901693059216_8301339564534923264_n.jpg
│   │   ├── ...
│   │   └── 92018365_2773046682923472_1923690185153839104_n.jpg
│   ├── test_set
│   │   ├── 000_0.png
│   │   └── ...
│   └── videos
│       ├── 000
│       │   ├── VIRB0391.MP4
│       │   └── ...
│       ├── ...
│       │   └── ...
│       └── 014
│           ├── VID_20180529_112517.mp4
│           └── ...
├── env
├── estensi
│   ├── painting_detection
│   │   ├── constants.py
│   │   ├── detection.py
│   │   ├── evaluation.py
│   │   └── utils.py
│   ├── painting_rectification
│   │   ├── rectification.py
│   │   └── utils.py
│   ├── painting_retrieval
│   │   ├── evaluation.py
│   │   ├── retrieval.py
│   │   └── utils.py
│   ├── people_detection
│   │   ├── cfg
│   │   │   └── yolov3.cfg
│   │   ├── darknet.py
│   │   ├── data
│   │   │   └── coco.names
│   │   ├── detection.py
│   │   ├── preprocess.py
│   │   ├── utils.py
│   │   └── yolov3.weights
│   ├── people_localization
│   │   ├── localization.py
│   │   └── utils.py
│   └── utils.py
├── estensi.py
├── painting_detection_evaluation.py
├── painting_retrieval_evaluation.py
├── README.md
└── requirements.txt
  ```

## Instructions
- Make sure to have installed all requirements (see `requirements.txt`).
- PyTorch requires a separate installation, depending from the system (CUDA version, CPU, etc.)
- Place the `dataset` folder at the same level as `estensi.py` and the `estensi` package.
- Place [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights) into `estensi/people_detection`.
- Run:
  ```bash
  estensi.py --video <path/to/video> [--include_steps] [--skip_frames]
  ```
  where:
  - `--video` targets the video in the dataset to analyze,
  - `--include_steps` tells the script to show useful debug information,
  - `--frame_skip` makes the script skip frames during analysis. 
- Alternatively run:
    ```bash
  estensi.py --folder </path/to/folder/> [--include_steps] [--skip_frames]
  ```
  where:
  - `--folder` targets the folder containing different videos to analyze,
  - `--include_steps` tells the script to show useful debug information,
  - `--frame_skip` makes the script skip frames during analysis. 

- While the video is playing, press the "R" button to start the painting retrieval, rectification and localization. You will see the outputs in new windows and in the command line for more details. Press any keyboard buttons to resume the video.
- While the video is playing, press the "P" button to pause the video, and any buttons to resume it.
- While the video is playing, press the "Q" button to quit the video.

## Evaluation
#### Painting detection
- The script will create a `test_set` folder under the `dataset` folder, containing the frames captured from a randomly selected set  of videos.
- Place the `ground_truth` folder under the `dataset` folder.
- If no argument is passed to the script, the test set will be evaluated with the system hyperparameters configuration.
Otherwise, it will be evaluated with the passed configuration.
- Run:
  ```bash
  painting_detection_evaluation.py [--param <param_grid_file_path>]
  ```
  where:
  - `--param` is the path of the json file containing the parameters grid for grid search evaluation.
#### Painting retrieval
- The script will create a `test_set` folder under the `dataset` folder, containing the frames captured from a randomly selected set  of videos.
- Place the `ground_truth` folder under the `dataset` folder.
- Run:
  ```bash
  painting_retrieval_evaluation.py --mode <mode_str> [--rank_scope <scope_int>]
  ```
  where:
  - `--mode` is the mode ['classification', 'retrieval'] in which the evaluation is done,
  - `--rank_scope` is the scope of the ranking list where a relevant item can be found. It will be ignored in classification mode.

