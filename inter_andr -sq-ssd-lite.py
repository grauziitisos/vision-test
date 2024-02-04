# test wrapper for https://github.com/qfgaohao/pytorch-ssd
#adb forward tcp:8080 tcp:8080
import datetime
MODEL_NAME_SUFFIX_IMG_NAMING="_sq-ssd-lite_01-09"
print(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, py file open...")
import os, requests
import numpy as np
import cv2
import torch
import asyncio
#import torchvision
import glob as glob
#import urllib
#import os
import time
import argparse
from torch import Tensor
#import yaml
#import matplotlib.pyplot as plt
###from PIL import Image
from torch import nn
###from torchvision import ops
###from torchvision.models.detection import FasterRCNN
###from torchvision.models.detection.rpn import AnchorGenerator

#CONFIG....
DELETE_IMAGES_AFTER_UPLOADED = False #True
#PATH_TO_WEIGHTS = "../../4AI_tikli_pet/2024-01-09/sq-ssd-lite/fi_dis_2024-01-09_01-51-14_sq-ssd-lite-Epoch-19-Loss-6.1730302174886065.pth"
#path_type='2'#"1"
PATH_TO_WEIGHTS = "../../4AI_tikli_pet/2023-12-10/sq-ssd/_fi__2023-12-10_02-51-57_sq-ssd-lite-Epoch-39-Loss-5.987395286560059.pth"
path_type="1"
PATH_TO_TEST_IMAGES = "__testim"
#DEFAULT_CAMERA_URL = 'http://127.0.0.1:8080/?action=stream'
DEFAULT_CAMERA_URL = 'https://cam1.jtag.me/?action=stream'
UPLOAD_URL = "https://jtag.me/tu.php"
LOG_URL = "https://jtag.me/tl.php"
DO_SKIP_TEST_IMAGE_FILES = False
CAMERA_TIME_LIMIT_SECONDS = 60*5

detection_threshold = 0.25
SAVE_TRESHOLD = 0.65
RESULT_IMAGES_PATH = "__result"


UPLOAD_EVERY_SECONDS = 2

TIME_HOLD_S = 3
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#cap, frame_width, frame_height = read_return_video_data(VIDEO_PATH)
#args ={"classes": None,
#       "track": False}
CLASSES_BEAUTY=["background", "Celjaziime"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES_BEAUTY), 3))

parser = argparse.ArgumentParser()
parser.add_argument(
    '-th', '--threshold', 
    default=0.3, 
    type=float,
    help='detection threshold'
)
parser.add_argument(
    '-s', '--show',  
    action='store_true',
    help='visualize output only if this argument is passed'
)

parser.add_argument(
    '-d', '--device', 
    default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    help='computation/training device, default is GPU if GPU present'
)
#sis ignored
RESIZE_TO = 640

parser.add_argument(
    '-nlb', '--no-labels',
    dest='no_labels',
    action='store_true',
    help='do not show labels during on top of bounding boxes'
)

parser.add_argument(
    '--classes',
    nargs='+',
    type=int,
    default=None,
    help='filter classes by visualization, --classes 1 2 3'
)

parser.add_argument(
    '-u', '--cam-url',
    type=str,
    default=DEFAULT_CAMERA_URL,
    help='camera stream url (includind query string as well)'
)

args = vars(parser.parse_args())


url = args['cam_url']
SHOW_PREVIEW = True#args['show']


#!CONFIG....
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIB includes from vision-transformers github~?LICENCE?CREDITS??
def collect_all_images(dir_test):
    """
    Function to return a list of image paths.
    :param dir_test: Directory containing images or single image path.
    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.gif']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images  
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIB includes: annotate
def inference_annotations_ssd(
    boxes, labels, probs,
    class_names, orig_image
):
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        #print(box)
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                (int(box[0]) + 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
    return orig_image

def convert_detections(
    outputs, 
    detection_threshold, 
    classes,
    args
):
    """
    Return the bounding boxes, scores, and classes.
    """
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()

    # Filter by classes if args.classes is not None.
    if args['classes'] is not None:
        labels = outputs[0]['labels'].cpu().numpy()
        lbl_mask = np.isin(labels, args['classes'])
        scores = scores[lbl_mask]
        mask = scores > detection_threshold
        draw_boxes = boxes[lbl_mask][mask]
        scores = scores[mask]
        labels = labels[lbl_mask][mask]
        pred_classes = [classes[i] for i in labels]
    # Else get outputs for all classes.
    else:
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicited class names.
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

    return draw_boxes, pred_classes, scores


def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):
        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )
        return img

def annotate_fps(orig_image, fps_text):
    draw_text(
        orig_image,
        f"FPS: {fps_text:0.1f}",
        pos=(20, 20),
        font_scale=1.0,
        text_color=(204, 85, 17),
        text_color_bg=(255, 255, 255),
        font_thickness=2,
    )
    return orig_image
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIB includes: annotate
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels
    
class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image

import collections
SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

import itertools
from typing import List
import math
def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # big sized square box
            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # change h/w ratio of the small sized box
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

class config:
    image_size = 300
    image_mean = np.array([127, 127, 127])  # RGB layout
    image_std = 128.0
    iou_threshold = 0.45
    center_variance = 0.1
    size_variance = 0.2

    specs = [
    SSDSpec(17, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]


    priors = generate_ssd_priors(specs, image_size)

class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            # I'm Afraid I Can't Do That, Dave
            # raise Exception(f"{key} is not in the clock.")
            print("I'm Afraid I Can't Do That, Dave")
            return time.time() - time.time()
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)



def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]



def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])



def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = self.device#torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        #print(boxes)
        #print (scores)
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

def create_squeezenet_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


def _no_grad_normal_(tensor, mean, std, generator=None):
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)

from typing import Optional as _Optional    
def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    r"""Fill the input Tensor with values drawn from the normal distribution.

    :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            normal_, (tensor,), tensor=tensor, mean=mean, std=std, generator=generator
        )
    return _no_grad_normal_(tensor, mean, std, generator)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalization
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

import warnings
def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: _Optional[torch.Generator] = None,
):
    r"""Fill the input `Tensor` with values using a Kaiming uniform distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            kaiming_uniform_,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
            generator=generator)

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)

def _no_grad_fill_(tensor, val):
    with torch.no_grad():
        return tensor.fill_(val)

def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fill the input Tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(constant_, (tensor,), tensor=tensor, val=val)
    return _no_grad_fill_(tensor, val)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    normal_(m.weight, mean=0.0, std=0.01)
                else:
                    kaiming_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    ###if pretrained:
    ###    model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\\_center * center_variance = \\frac {real\\_center - prior\\_center} {prior\\_hw}$$
        $$exp(predicted\\_hw * size_variance) = \\frac {real\\_hw} {prior\\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)

def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], locations.dim() - 1) 


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

from typing import List, Tuple
from collections import namedtuple
import torch.nn.functional as F
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #
class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)



from torch.nn import Conv2d, Sequential, ModuleList, ReLU
def create_squeezenet_ssd_lite(num_classes, is_test=False):
    base_net = squeezenet1_1(False).features  # disable dropout layer

    source_layer_indexes = [
        12
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2),
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


net = create_squeezenet_ssd_lite(len(CLASSES_BEAUTY), is_test=True)

if(path_type=='1'):
    print("loading state dict...")
    net.load_state_dict(torch.load(PATH_TO_WEIGHTS))
elif(path_type=='2'):
    print("loading parallel state dict...")
    os.environ.setdefault('nproc-per-node','1')
    os.environ.setdefault('max-restarts','9')
    os.environ.setdefault('rdzv-id','3')
    os.environ.setdefault('rdzv-backend','static')
    os.environ.setdefault('rdzv-endpoint','127.0.0.10:7000')
    os.environ.setdefault('nnodes','1')
    os.environ.setdefault('LOCAL_RANK','0')
    os.environ.setdefault('RANK','0')
    os.environ.setdefault('WORLD_SIZE','1')
    os.environ.setdefault('MASTER_ADDR','127.0.0.10')
    os.environ.setdefault('MASTER_PORT','7000')
    torch.distributed.init_process_group(backend="gloo")
    net = torch.nn.parallel.DistributedDataParallel(net,  device_ids=None,
                                                                      output_device=None)
    net.load_state_dict(
                       torch.load(PATH_TO_WEIGHTS))#
else:
    print("loading net")
    net.load(PATH_TO_WEIGHTS)

predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
#model.eval()


frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
LASTFRAME = None

UPLOAD_RUNNING_FLAG = False
def upload():
    global UPLOAD_RUNNING_FLAG
    UPLOAD_RUNNING_FLAG = True
    if not os.path.exists(RESULT_IMAGES_PATH): 
        os.makedirs(RESULT_IMAGES_PATH)
    for fn in os.listdir(RESULT_IMAGES_PATH):
        if fn.endswith('.jpg'):
            try:
                to_delete = False
                with open(RESULT_IMAGES_PATH+"/"+fn, 'rb') as f:
                    response = requests.post(UPLOAD_URL, files={"f": (fn, f, 'image/jpeg')})
                    if(response):
                        if(response.text == fn):
                            to_delete = True
                        print(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} OK .:. {response.text}")
                if(to_delete):
                    if(DELETE_IMAGES_AFTER_UPLOADED):
                        os.remove(RESULT_IMAGES_PATH+"/"+fn)
                    else:
                        if not os.path.exists(RESULT_IMAGES_PATH+"/"+"d"): 
                            os.makedirs(RESULT_IMAGES_PATH+"/"+"d")
                        os.rename(RESULT_IMAGES_PATH+"/"+fn, RESULT_IMAGES_PATH+"/"+"d"+"/"+fn)
            finally:
                ...
    UPLOAD_RUNNING_FLAG = False


#async 
def recognize(img):
                global LASTFRAME, last_time, total_fps, frame_count, DO_BREAK, SHOW_PREVIEW, CLASSES, CLASSES_BEAUTY, COLORS
                # GIL vēl aizvien ir te :(*)
                ##await asyncio.sleep(0)
                #šie ir realtime.
                #LASTFRAME = jpg
                #return
                start_time_ms = time.time_ns() // 1_000_000

                start_forward_ms = time.time_ns() // 1_000_000
                jpg = img.copy()
                #if jpg.shape[2] == 1:
                #    jpg = cv2.cvtColor(jpg, cv2.COLOR_GRAY2RGB)
                #else:
                #    jpg = cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    # Get predictions for the current frame.
                    boxes, labels, probs  = predictor.predict(jpg, 10, detection_threshold)#model.forward(image.to(DEVICE))
                forward_end_time_ms = time.time_ns() // 1_000_000

                forward_pass_time_ms = forward_end_time_ms - start_forward_ms
                forward_pass_time = forward_pass_time_ms/1000
                # Get the current fps.
                fps = 1 / (forward_pass_time)
                # Add `fps` to `total_fps`.
                total_fps += fps
                # Increment frame count.
                frame_count += 1

                # Carry further only if there are detected boxes.
                had_match = 0.00
                if len(boxes) != 0:   
                    jpg = inference_annotations_ssd(
                        boxes, labels, probs,
    CLASSES_BEAUTY, jpg
                    )
                    for i in probs:
                         if (i.item()>SAVE_TRESHOLD):
                            had_match = i.item()
                            if not os.path.exists(RESULT_IMAGES_PATH): 
                                os.makedirs(RESULT_IMAGES_PATH)
                            cv2.imwrite(RESULT_IMAGES_PATH+"/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+MODEL_NAME_SUFFIX_IMG_NAMING+".jpg", jpg)
                            break
                         
                LASTFRAME = annotate_fps(jpg, fps)
                final_end_time_ms = time.time_ns() // 1_000_000
                forward_and_annot_time = (final_end_time_ms - start_time_ms)/1000
                print_string = f"Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, Frame: {frame_count}, Forward pass FPS: {fps:.3f}, "
                print_string += f"Forward pass time: {forward_pass_time:.3f} seconds, "
                print_string += f"Forward pass + annotation time: {forward_and_annot_time:.3f} seconds"
                log_dhr = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {had_match:.3f} "
                log_print(log_dhr+print_string, 
                          log_dhr,
                          f"{print_string}\n")
                if(SHOW_PREVIEW):
                    cv2.imshow('Prediction', LASTFRAME)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        DO_BREAK = True
                last_time = time.time()        

last_time = time.time()        
DO_BREAK = False
cap = None

async def connect_camera():
    global cap
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, seeking the camera...")
    retr_count = 0
    while(True):
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print_string = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, Error: Could not open MJPEG stream. waiting 1s to retry."
                if(retr_count <3):
                    print(print_string)
                    retr_count+= 1
                else:
                    log_print(print_string, print_string, "")
                    retr_count = 0
                time.sleep(1)
                continue
            break
        except Exception as ex:
            print(ex)
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, waiting 1s to retry.")
            time.sleep(1)
            ...

#async
def log_print(print_string, log_prefix, log_string):
    print(print_string)
    try:
        requests.post(LOG_URL, data={"tx": f"{log_prefix}{log_string}\n"})
    finally:
        ...

CAMERA_BEGIN_TIME = time.time()
LAST_CAP_FRAME = None
async def cameraloop():
    global LAST_CAP_FRAME, cap, CAMERA_BEGIN_TIME, CAMERA_TIME_LIMIT_SECONDS
    while(True):
        if(time.time()-CAMERA_BEGIN_TIME +3 >CAMERA_TIME_LIMIT_SECONDS): return
        #print("cameraloop")
        # GIL vēl aizvien ir te :(*)
        await asyncio.sleep(0)
        success, jpg = cap.read()
        if not success:
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, could not read frame.")
            time.sleep(1)
            await connect_camera()
            continue
        else:
             LAST_CAP_FRAME = jpg


last_upload_tries_counter = 0
async def upload_loop():
    global last_upload_tries_counter, CAMERA_BEGIN_TIME, CAMERA_TIME_LIMIT_SECONDS
    last_time = time.time()
    wtfevloop = asyncio.get_event_loop()
    while(True):
         if(time.time()-CAMERA_BEGIN_TIME +3 >CAMERA_TIME_LIMIT_SECONDS):
            if not os.path.exists(RESULT_IMAGES_PATH): 
                os.makedirs(RESULT_IMAGES_PATH)
            HAD_jpg_FLAG = False
            for fn in os.listdir(RESULT_IMAGES_PATH):
                if fn.endswith('.jpg'):
                    HAD_jpg_FLAG = True
                    break
            if(not HAD_jpg_FLAG):
                return
            else:
                if last_upload_tries_counter > 3: return
                last_upload_tries_counter +=1
         if(not UPLOAD_RUNNING_FLAG):
            if(time.time() - last_time >UPLOAD_EVERY_SECONDS):
                await wtfevloop.run_in_executor(None, upload)
                

async def recognize_loop():
     global LAST_CAP_FRAME
     wtfevloop = asyncio.get_event_loop()
     while(True):
        if(time.time()-CAMERA_BEGIN_TIME +3 >CAMERA_TIME_LIMIT_SECONDS): return
        if(LAST_CAP_FRAME is None):
            await asyncio.sleep(1)
            continue
        await wtfevloop.run_in_executor(None, recognize, LAST_CAP_FRAME)

async def the_loop():
    global last_time, total_fps, frame_count, DO_BREAK, LAST_CAP_FRAME, CAMERA_BEGIN_TIME
    # first, try test images + metric
    ims = collect_all_images(PATH_TO_TEST_IMAGES)
    if(ims[0]==PATH_TO_TEST_IMAGES or DO_SKIP_TEST_IMAGE_FILES):
        ...
    else:
        for i in ims:
            CAMERA_BEGIN_TIME = time.time()
            #recognize(cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB))
            recognize(cv2.imread(i))
        avg_fps = total_fps / frame_count
        print_string = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, Average FFPS: {avg_fps:.3f}"
        log_print( print_string,
                          print_string,
                          f"\n")
        avg_fps = 0.0
        total_fps = 0
        frame_count = 0
        #nomainot src uzkarās...
        cv2.destroyAllWindows()
    #! testun
    last_time = time.time()
    print_string = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Camera start .:. "
    log_print( print_string,
                          print_string,
                          f"\n")
    await connect_camera()
    CAMERA_BEGIN_TIME = time.time()

    # TODO: FPPS VS RECPSC
    # FPPS SKAITA., DELAY TIME() UN TAD NĀKAMO SĀK ATPAZĪT JA IOR.

    # Use asyncio.gather to run tasks concurrently
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, launching loops...")
    tasks = [
        cameraloop(),
        recognize_loop(),
        upload_loop()
    ]
    await asyncio.gather(*tasks)


    # Close all frames and video windows.
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / (frame_count if frame_count != 0 else 1)
    print_string = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, Average FPS: {avg_fps:.3f}"
    log_print( print_string,
                          print_string,
                          f"\n")
    print_string = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Camera end.."
    log_print( print_string,
                          print_string,
                          f"\n")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Mdl: {MODEL_NAME_SUFFIX_IMG_NAMING}, py file exit ok...")

#asyncio.run(the_loop())
#evloop.run_until_complete(
asyncio.run(the_loop())
     #)