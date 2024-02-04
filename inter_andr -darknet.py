#adb forward tcp:8080 tcp:8080
import datetime
MODEL_NAME_SUFFIX_IMG_NAMING="_mini_darknet_01-14"
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
from PIL import Image
from torch import nn
from torchvision import ops
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

#CONFIG....
DELETE_IMAGES_AFTER_UPLOADED = False #True
PATH_TO_WEIGHTS = "../../4AI_tikli_pet/2024-01-14/sign_mini_darknet/best_model.pth"
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
SHOW_PREVIEW = args['show']


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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIB INCLUDES :: image transforms
def resize(im, img_size=640, square=False):
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im



class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>
             transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            #_log_api_usage_once(self)
            ...
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
    
from typing import Any
@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
        return isinstance(img, Image.Image)

@torch.jit.unused
def get_image_num_channels(img: Any) -> int:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            return len(img.getbands())
        else:
            return img.channels
    raise TypeError(f"Unexpected type {type(img)}")

@torch.jit.unused
def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)
    
@torch.jit.unused
def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}
   
import sys
def to_tensor(pic) -> Tensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        #_log_api_usage_once(to_tensor)
        ...
    if not (_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype).div(255)
        else:
            return img

    #if accimage is not None and isinstance(pic, accimage.Image):
    #    nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
    #    pic.copyto(nppic)
    #    return torch.from_numpy(nppic).to(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {"I": np.int32, "I;16" if sys.byteorder == "little" else "I;16B": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

    if pic.mode == "1":
        img = 255 * img
    img = img.view(pic.size[1], pic.size[0], get_image_num_channels(pic))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype).div(255)
    else:
        return img



class ToTensor:
    """Convert a PIL Image or ndarray to tensor and scale the values accordingly.

    This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __init__(self) -> None:
        #_log_api_usage_once(self)
        ...

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image. This function does not support torchscript.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        #_log_api_usage_once(to_pil_image)
        ...

    if isinstance(pic, torch.Tensor):
        if pic.ndim == 3:
            pic = pic.permute((1, 2, 0))
        pic = pic.numpy(force=True)
    elif not isinstance(pic, np.ndarray):
        raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

    if pic.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        pic = np.expand_dims(pic, 2)
    if pic.ndim != 3:
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    if pic.shape[-1] > 4:
        raise ValueError(f"pic should not have > 4 channels. Got {pic.shape[-1]} channels.")

    npimg = pic

    if np.issubdtype(npimg.dtype, np.floating) and mode != "F":
        npimg = (npimg * 255).astype(np.uint8)

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        elif npimg.dtype == np.int16:
            expected_mode = "I;16" if sys.byteorder == "little" else "I;16B"
        elif npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}")
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ["LA"]
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(f"Only modes {permitted_2_channel_modes} are supported for 2D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "LA"

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError(f"Input type {npimg.dtype} is not supported")

    return Image.fromarray(npimg, mode=mode)

class ToPILImage:
    """Convert a tensor or an ndarray to PIL Image

    This transform does not support torchscript.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while adjusting the value range depending on the ``mode``.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:

            - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
            - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``, ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """

    def __init__(self, mode=None):
        #_log_api_usage_once(self)
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return to_pil_image(pic, self.mode)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += f"mode={self.mode}"
        format_string += ")"
        return format_string


def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = Compose([
        ToPILImage(),
        ToTensor(),
    ])
    return transform(image)


#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIB includes: image transforms

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIB includes: annotate
def inference_annotations(
    draw_boxes, 
    pred_classes, 
    scores, 
    classes,
    classes_beautifule_named,
    colors, 
    orig_image, 
    image, 
    args
):
    height, width, _ = orig_image.shape
    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    
    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1 = (int(box[0]/image.shape[1]*width), int(box[1]/image.shape[0]*height))
        p2 = (int(box[2]/image.shape[1]*width), int(box[3]/image.shape[0]*height))
        class_name = classes_beautifule_named[classes.index(pred_classes[j])]
        #if args['track']:
        #    color = colors[classes.index(' '.join(class_name.split(' ')[1:]))]
        #else:
        if(True):
            color = colors[classes.index(pred_classes[j])]
        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color, 
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        if True:#not args['no_labels']:
            # For filled rectangle.
            final_label = class_name + ' ' + str(round(scores[j], 2))
            w, h = cv2.getTextSize(
                final_label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw / 3, 
                thickness=tf
            )[0]  # text width, height
            w = int(w - (0.20 * w))
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(
                orig_image, 
                p1, 
                p2, 
                color=color, 
                thickness=-1, 
                lineType=cv2.LINE_AA
            )  
            cv2.putText(
                orig_image, 
                final_label, 
                (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw / 3.8, 
                color=(255, 255, 255), 
                thickness=tf, 
                lineType=cv2.LINE_AA
            )
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIB includes: model
# A DarkNet model with reduced output channels for each layer.

class DarkNet(nn.Module):
    def __init__(self, initialize_weights=True, num_classes=1000):
        super(DarkNet, self).__init__()

        self.num_classes = num_classes
        self.features = self._create_conv_layers()
        self.pool = self._pool()
        self.fcs = self._create_fc_layers()

        if initialize_weights:
            # Random initialization of the weights
            # just like the original paper.
            self._initialize_weights()

    def _create_conv_layers(self):
        conv_layers = nn.Sequential(
            nn.Conv2d(3, 4, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(4, 8, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return conv_layers

    def _pool(self):
        pool = nn.Sequential(
            nn.AvgPool2d(7),
        )
        return pool
    
    def _create_fc_layers(self):
        fc_layers = nn.Sequential(
            nn.Linear(128, self.num_classes)
        )
        return fc_layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.fcs(x)
        return x
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LIB includes: model

checkpoint = torch.load(PATH_TO_WEIGHTS, map_location=DEVICE)
NUM_CLASSES = checkpoint['data']['NC']
CLASSES = checkpoint['data']['CLASSES']
#build_model = create_model["fasterrcnn_mini_darknet"]

    # Load the Mini DarkNet model features.
backbone = DarkNet(num_classes=10).features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 128 for this custom Mini DarkNet model.
backbone.out_channels = 128

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
roi_pooler = ops.MultiScaleRoIAlign(#torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
model = FasterRCNN(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
#build_model(num_classes=NUM_CLASSES, coco_model=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
evloop = asyncio.get_event_loop()
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
def recognize(jpg):
                global LASTFRAME, last_time, total_fps, frame_count, DO_BREAK, SHOW_PREVIEW, CLASSES, CLASSES_BEAUTY, COLORS
                # GIL vēl aizvien ir te :(*)
                ##await asyncio.sleep(0)
                #šie ir realtime.
                #LASTFRAME = jpg
                #return
                start_time_ms = time.time_ns() // 1_000_000
                orig_frame = jpg.copy()
                frame = resize(jpg, 640, square=True)
                image = frame.copy()
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = infer_transforms(image)
                # Add batch dimension.
                image = torch.unsqueeze(image, 0)
                # Get the start time.
                start_forward_ms = time.time_ns() // 1_000_000
                with torch.no_grad():
                    # Get predictions for the current frame.
                    outputs = model(image.to(DEVICE))
                forward_end_time_ms = time.time_ns() // 1_000_000

                forward_pass_time_ms = forward_end_time_ms - start_forward_ms
                forward_pass_time = forward_pass_time_ms/1000
                # Get the current fps.
                fps = 1 / (forward_pass_time)
                # Add `fps` to `total_fps`.
                total_fps += fps
                # Increment frame count.
                frame_count += 1

                # Load all detection to CPU for further operations.
                outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

                # Carry further only if there are detected boxes.
                had_match = 0.00
                if len(outputs[0]['boxes']) != 0:
                    draw_boxes, pred_classes, scores = convert_detections(
                        outputs, detection_threshold, CLASSES, args
                    )
                    
                    frame = inference_annotations(
                        draw_boxes, 
                        pred_classes, 
                        scores,
                        CLASSES, 
                        CLASSES_BEAUTY,
                        COLORS, 
                        orig_frame, 
                        frame,
                        args
                    )
                    for i in scores:
                         if (i>SAVE_TRESHOLD):
                            had_match = i
                            if not os.path.exists(RESULT_IMAGES_PATH): 
                                os.makedirs(RESULT_IMAGES_PATH)
                            cv2.imwrite(RESULT_IMAGES_PATH+"/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+MODEL_NAME_SUFFIX_IMG_NAMING+".jpg", frame)
                            break
                else:
                    frame = orig_frame
                LASTFRAME = annotate_fps(frame, fps)
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
     global LAST_CAP_FRAME, evloop, CAMERA_BEGIN_TIME, CAMERA_TIME_LIMIT_SECONDS
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