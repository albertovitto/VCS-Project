import time

import numpy as np
import cv2

from Luca.vcsp.painting_retrieval.retrieval import PaintingRetrieval
from Luca.vcsp.painting_detection.detection import get_bb
from Luca.vcsp.utils.drawing import draw_bb
from Luca.vcsp.utils.multiple_show import show_on_row

import torch
from torch.autograd import Variable
import random
from Cristian.YOLOv3.util import *
from Cristian.YOLOv3.darknet import Darknet
from Cristian.YOLOv3.preprocess import prep_image, inp_to_image, letterbox_image


class Yolo():
    def __init__(self, confidence=0.5, nms_thesh=0.4, num_classes=80):
        self.confidence = confidence
        self.nms_thesh = nms_thesh
        self.CUDA = torch.cuda.is_available()
        self.num_classes = num_classes
        self.classes = load_classes('data/coco.names')
        self.bbox_attrs = 5 + self.num_classes
        print("Loading network.....")
        self.model = Darknet('cfg/yolov3.cfg')
        self.model.load_weights('yolov3.weights')
        print("Network successfully loaded")
        self.model.net_info["height"] = "416"
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32
        if self.CUDA:
            self.model.cuda()
        self.model(self.get_test_input(self.inp_dim, self.CUDA), self.CUDA)
        self.model.eval()

    @staticmethod
    def get_test_input(input_dim, CUDA):
        img = cv2.imread("dog-cycle-car.png")
        img = cv2.resize(img, (input_dim, input_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1))
        img_ = img_[np.newaxis, :, :, :] / 255.0
        img_ = torch.from_numpy(img_).float()
        img_ = Variable(img_)
        if CUDA:
            img_ = img_.cuda()
        return img_

    @staticmethod
    def prep_image(img, inp_dim):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """
        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    def retrieve_people_bb(self, frame):
        img, orig_im, dim = self.prep_image(frame, self.inp_dim)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = self.model(Variable(img), self.CUDA)

        if type(output) == int:
            return [(0, 0, 0, 0)]

        output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(self.inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        bb = []
        for x in output:
            if int(x[7].item()) != 0:
                continue
            x1 = int(x[1].item())
            y1 = int(x[2].item())
            x2 = int(x[3].item())
            y2 = int(x[4].item())
            w = x2 - x1
            h = y2 - y1
            bb.append((x1, y1, w, h))

        return bb


def intersection_over_person(painting, person):
    # inspired by https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    A = painting
    B = person

    # top-left coord of box A/B
    x1a, y1a, wa, ha = A
    x1b, y1b, wb, hb = B

    # bottom-right coord of box A/B
    x2a = x1a + wa
    y2a = y1a + ha
    x2b = x1b + wb
    y2b = y1b + hb

    # top-left coord of intersection
    x1i = max(x1a, x1b)
    y1i = max(y1a, y1b)

    # bottom-right coord og intersection
    x2i = min(x2a, x2b)
    y2i = min(y2a, y2b)

    areaI = abs(max((x2i - x1i, 0)) * max((y2i - y1i), 0))
    areaB = wb * hb

    if areaB == 0:
        return 0
    return areaI / float(areaB)


def check_bbs_iou(paintings, people, threshold=0.7):
    selected_people = []
    for person in people:
        selected_people.append(person)
        for painting in paintings:
            iop = intersection_over_person(painting, person)
            if iop > threshold:
                selected_people.pop()
                break
    return selected_people


if __name__ == '__main__':

    db_dir_path = '../../dataset/paintings_db/'
    files_dir_path = '../../dataset/'

    retrieval = PaintingRetrieval(db_dir_path, files_dir_path)
    retrieval.train()

    # video_name = '000/VIRB0393.MP4'
    # video_name = '001/GOPR5826.MP4'
    # video_name = '005/GOPR2045.MP4'
    # video_name = '012/IMG_4086.MOV'
    # video_name = '005/GOPR2051.MP4'
    # video_name = '004/IMG_3803.MOV'
    # video_name = '008/VIRB0419.MP4'
    # video_name = '008/VIRB0427.MP4'
    # video_name = '012/IMG_4080.MOV'
    video_name = '002/20180206_114720.mp4'

    video_path = '../../dataset/videos/%s' % video_name

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: video not opened correctly")

    lost_frames = 0
    pos_frames = 0
    skip_frames = True

    yolo = Yolo()

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            output, rois, bbs = get_bb(frame, include_steps=False)
            people_bb = yolo.retrieve_people_bb(frame)
            people_bb = check_bbs_iou(paintings=bbs, people=people_bb)
            # print(people_bb)
            for bb in people_bb:
                x, y, w, h = bb
                draw_bb(output, tl=(x, y), br=(x + w, y + h), color=(0, 0, 255), label="person")
            cv2.imshow("Painting detection", output)

            key = cv2.waitKey(1)
            if key == ord('q'):  # quit
                break
            if key == ord('p'):  # pause
                cv2.waitKey(-1)
            if key == ord('r'):  # show rois with image retrieval
                for i, roi in enumerate(rois):
                    rank, _ = retrieval.predict(roi)
                    cv2.putText(roi, "{}".format(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3,
                                False)
                    print("Roi {} - rank = {}".format(i, rank))
                    ground_truth = cv2.imread('../../dataset/paintings_db/' + "{:03d}.png".format(rank[0]))
                    cv2.imshow("Roi {}".format(i), show_on_row(roi, ground_truth))
                cv2.waitKey(-1)
                for i, roi in enumerate(rois):
                    cv2.destroyWindow("Roi {}".format(i))

            if skip_frames:
                pos_frames += video.get(cv2.CAP_PROP_FPS)
                video.set(cv2.CAP_PROP_POS_FRAMES, pos_frames)
        else:
            lost_frames += 1
            if lost_frames > 10:
                print("Too many errors reading video or video ended")
                break

    video.release()
    cv2.destroyAllWindows()
