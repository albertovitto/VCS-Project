import os

from estensi.people_detection.utils import *
from estensi.people_detection.darknet import Darknet
from estensi.people_detection.preprocess import letterbox_image


class Yolo():
    def __init__(self, confidence=0.5, nms_thesh=0.4, num_classes=80, yolo_path=""):
        self.confidence = confidence
        self.nms_thesh = nms_thesh
        self.CUDA = torch.cuda.is_available()
        self.num_classes = num_classes
        self.classes = load_classes(os.path.join(yolo_path, 'data', 'coco.names'))
        self.bbox_attrs = 5 + self.num_classes
        print("Loading network.....")
        self.model = Darknet(os.path.join(yolo_path, 'cfg', 'yolov3.cfg'))
        self.model.load_weights(os.path.join(yolo_path, 'yolov3.weights'))
        print("Network successfully loaded")
        self.model.net_info["height"] = "416"
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32
        if self.CUDA:
            self.model.cuda()
        self.model(self.get_test_input(self.inp_dim, self.CUDA, yolo_path), self.CUDA)
        self.model.eval()

        # people detection correction
        self.prev_ok_bbs = []
        self.prev_ko_bbs_buffer = []

    @staticmethod
    def get_test_input(input_dim, CUDA, path):
        img = cv2.imread(os.path.join(path, "dog-cycle-car.png"))
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

    @staticmethod
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

    def check_bbs_iop(self, paintings, people):
        selected_people = []
        discarded_people = []
        for person in people:
            selected_people.append(person)
            for painting in paintings:
                iop = self.intersection_over_person(painting, person)
                if iop == 1:
                    discarded_people.append(selected_people.pop())
                    break
        return selected_people, discarded_people

    @staticmethod
    def remove_useless_detections(people_bbs):
        return list(filter(lambda bb: bb != (0, 0, 0, 0), people_bbs))

    def discard_bbs(self, ok_bbs, ko_bbs, threshold=100):
        # - considerare le people detection nel frame precedente e le people detection scartate nel frame precedente,
        #   per poi eliminare le detection nel frame corrente che sono vicine a quelle che prima abbiamo scartato.
        #   Questo metodo si basa sull'assunzione che la camera non si muova troppo velocemente da un frame all'altro.
        #   I problemi sono due: la camera a volte si muove velocemente; non possiamo skippare i frame.
        # - fare l'opposto, ovvero considerare le painting detection del/dei frame precedenti e scartare le people detection
        #   del frame corrente se si overlappano non solo col frame corrente, ma anche con/coi precedenti.
        #   Però abbiamo gli stessi problemi di prima.
        def center(bb):
            x, y, w, h = bb
            return np.asarray((x + w // 2, y + h // 2))

        okok_bbs = []
        for ok in ok_bbs:
            for ko_list in self.prev_ko_bbs_buffer:
                distances = np.asarray([np.linalg.norm(center(ok) - center(ko)) for ko in ko_list])
                if np.any(distances <= threshold):
                    ko_bbs.append(ok)
                else:
                    okok_bbs.append(ok)

        return okok_bbs, ko_bbs

    def get_people_bb(self, frame, painting_bbs=None):
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

        bbs = []
        for x in output:
            if int(x[7].item()) != 0:
                continue
            if torch.isnan(x).any():  # sometimes x has NaN values
                continue
            x1 = int(x[1].item())
            y1 = int(x[2].item())
            x2 = int(x[3].item())
            y2 = int(x[4].item())
            w = x2 - x1
            h = y2 - y1
            bbs.append((x1, y1, w, h))

        if painting_bbs is not None:
            ok_bbs, ko_bbs = self.check_bbs_iop(paintings=painting_bbs, people=bbs)
            ok_bbs = self.remove_useless_detections(ok_bbs)
            ok_bbs, ko_bbs = self.discard_bbs(ok_bbs, ko_bbs, threshold=400)
            self.prev_ok_bbs = ok_bbs
            self.prev_ko_bbs_buffer.append(ko_bbs)
            windows_size = 2
            while len(self.prev_ko_bbs_buffer) > windows_size:
                self.prev_ko_bbs_buffer.pop(0)
            return ok_bbs
        else:
            self.prev_ok_bbs = []
            return bbs
