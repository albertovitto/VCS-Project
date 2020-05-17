import numpy as np
import cv2
import torch
import pickle
import random

from Luca.vcsp.people_detection.YOLOv3.darknet import Darknet
from Luca.vcsp.people_detection.YOLOv3.preprocess import letterbox_image
from Luca.vcsp.people_detection.YOLOv3.util import write_results, load_classes
from Luca.vcsp.utils.drawing import draw_bb


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


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    if cls != 0:  # not PERSON
        return img

    area_bb = (c2[0] - c1[0]) * (c2[1] - c1[1])
    if area_bb == 0:
        return img

    img = draw_bb(img, tl=c1, br=c2, color=(0, 255, 0), label="person")
    return img


class PeopleDetector:

    def __init__(self):
        self.confidence = 0.5
        self.nms_thesh = float(0.4)

        self.CUDA = torch.cuda.is_available()

        self.num_classes = 80  # 2
        self.classes = load_classes('../vcsp/people_detection/YOLOv3/data/coco.names')

        self.bbox_attrs = 5 + self.num_classes

        print("Loading network.....")
        self.model = Darknet("../vcsp/people_detection/YOLOv3/cfg/yolov3.cfg")
        self.model.load_weights("../vcsp/people_detection/YOLOv3/yolov3.weights")
        print("Network successfully loaded")

        self.model.net_info["height"] = 416
        self.inp_dim = int(self.model.net_info["height"])

        if self.CUDA:
            self.model.cuda()

        self.model.eval()

    def get_bb(self, frame):
        img, orig_im, dim = prep_image(frame, self.inp_dim)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = self.model(torch.autograd.Variable(img), self.CUDA)
        output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)

        if type(output) == int:
            return orig_im

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(self.inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        for detect_obj in output:
            tl = tuple(detect_obj[1:3].int())
            br = tuple(detect_obj[3:5].int())

            cls = int(detect_obj[-1])
            if cls != 0:  # not PERSON
                continue

            area_bb = (br[0] - tl[0]) * (br[1] - tl[1])
            if area_bb == 0:
                continue

            draw_bb(orig_im, tl, br, color=(0, 255, 0), label="person")

        return orig_im

