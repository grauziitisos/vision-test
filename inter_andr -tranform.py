# test wrapper for https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
#adb forward tcp:8080 tcp:8080
import datetime
MODEL_NAME_SUFFIX_IMG_NAMING="_mobilevit_XXS_01-30"
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

import torchvision
import torch.nn as nn
import sys
import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
#import vision_transformers


#CONFIG....
DELETE_IMAGES_AFTER_UPLOADED = False #True
PATH_TO_WEIGHTS = "bin/2024-01-30/mobilevit_xxs/last_model_state.pth"
PATH_TO_TEST_IMAGES = "__testim"
#DEFAULT_CAMERA_URL = 'http://127.0.0.1:8080/?action=stream'
DEFAULT_CAMERA_URL = 'https://cam1.jtag.me/?action=stream'
UPLOAD_URL = "https://jtag.me/tu.php"
LOG_URL = "https://jtag.me/tl.php"
DO_SKIP_TEST_IMAGE_FILES = False#True #False
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
from argparse import Namespace

class MHSA(nn.Module):
    """
        Multi-head self attention: https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 attn_dropout = 0.0, 
                 bias = True
                 ):
        """
            :param embed_dim: embedding dimension
            :param num_heads: number of attention heads
            :param attn_dropout: attention dropout
            :param bias: use bias or not
        """
        super(MHSA, self).__init__()
        assert embed_dim % num_heads == 0, "Got: embed_dim={} and num_heads={}".format(embed_dim, num_heads)

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3*embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [B x N x C]
        b_sz, n_patches, _ = x.shape

        # linear projection to qkv
        # [B x N x C] --> [B x N x 3 x h x C]
        qkv = (self.qkv_proj(x).reshape(b_sz, n_patches, 3, self.num_heads, -1))
        # [B x N x 3 x h x C] --> [B x h x 3 x N x C]
        qkv = qkv.transpose(1, 3)
        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q * self.scaling
        # [B x h x N x C] --> [B x h x c x N]
        k = k.transpose(2, 3)

        # compute attention score
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = torch.matmul(q, k)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [B x h x N x N] x [B x h x N x c] --> [B x h x N x c]
        out = torch.matmul(attn, v)
        # [B x h x N x c] --> [B x N x h x c] --> [B x N x C=ch]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out

class TransformerEncoder(nn.Module):
    """
        Transfomer Encoder
    """
    def __init__(self, 
                 opts, 
                 embed_dim, 
                 ffn_latent_dim, 
                 num_heads = 8, 
                 attn_dropout = 0.0,
                 dropout = 0.1, 
                 ffn_dropout = 0.0,
                 transformer_norm_layer = "layer_norm",
                 ):
        """
            :param opts: arguments
            :param embed_dim: embedding dimension
            :param ffn_latent_dim: latent dimension of feedforward layer
            :param num_heads: Number of attention heads
            :param attn_dropout: attention dropout rate
            :param dropout: dropout rate
            :param ffn_dropout: feedforward dropout rate
            :param transformer_norm_layer: transformer norm layer
        """
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            MHSA(
                embed_dim, 
                num_heads, 
                # dim_head=embed_dim//num_heads,
                attn_dropout=attn_dropout
            ),
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            self.build_act_layer(opts=opts),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )

    @staticmethod
    def build_act_layer(opts):
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(act_type=act_type, inplace=inplace, negative_slope=neg_slope,
                                      num_parameters=1)
        return act_layer

    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


import torch.nn.functional as F
class MobileViTBlock(nn.Module):
    """
        MobileViT block: https://arxiv.org/pdf/2110.02178
    """
    def __init__(self, 
                 opts, 
                 in_channels, 
                 transformer_dim, 
                 ffn_dim,
                 n_transformer_blocks = 2,
                 head_dim = 32, 
                 attn_dropout = 0.1,
                 dropout = 0.1, 
                 ffn_dropout = 0.1, 
                 patch_h = 8,
                 patch_w = 8, 
                 transformer_norm_layer = "layer_norm",
                 conv_ksize = 3,
                 dilation = 1, 
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param transformer_dim: dimension of transformer encoder
            :param ffn_dim: dimension of feedforward layer
            :param n_transformer_block: number of transformer blocks
            :param head_dim: transformer head dimension     
            :param attn_dropout: Attention dropout     
            :param dropout: dropout
            :param ffn_dropout: feedforward dropout
            :param patch_h: split patch height size      
            :param patch_w: split patch width size
            :param transformer_norm_layer: transformer norm layer    
            :param conv_ksize: kernel size for convolutional block    
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.    
        """

        conv_3x3_in = ConvBlock(
            opts=opts, in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation
        )
        conv_1x1_in = ConvBlock(
            opts=opts, in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False
        )

        conv_1x1_out = ConvBlock(
            opts=opts, in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True
        )
        conv_3x3_out = ConvBlock(
            opts=opts, in_channels=2 * in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True
        )
        
        super(MobileViTBlock, self).__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(
                opts=opts, 
                embed_dim=transformer_dim, 
                ffn_latent_dim=ffn_dims[block_idx], 
                num_heads=num_heads,
                attn_dropout=attn_dropout, 
                dropout=dropout, 
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=transformer_norm_layer
            )
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, _, _, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x):
        res = x
        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)
        # learn global representations
        patches = self.global_rep(patches)
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)
        fm = self.fusion(torch.cat((res, fm), dim=1))

        return fm

class Identity(nn.Module):
    def __init__(self):
        """
            Identity operator
        """
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
SUPPORTED_NORM_FNS = [
    'batch_norm_2d', 
    'batch_norm_1d', 
    'sync_batch_norm', 
    'group_norm',
    'instance_norm_2d', 
    'instance_norm_1d',
    'layer_norm',
    'identity'
    ]

import math    
def get_normalization_layer(opts, num_features, norm_type = None, num_groups = None):

    norm_type = getattr(opts, "model.normalization.name", "batch_norm_2d") if norm_type is None else norm_type
    num_groups = getattr(opts, "model.normalization.groups", 1) if num_groups is None else num_groups
    momentum = getattr(opts, "model.normalization.momentum", 0.1)

    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None
    if norm_type == 'batch_norm_2d':
        norm_layer = nn.BatchNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == 'batch_norm_1d':
        norm_layer = nn.BatchNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type == 'sync_batch_norm':
        norm_layer = nn.SyncBatchNorm(num_features=num_features, momentum=momentum)
    elif norm_type == 'group_norm':
        num_groups = math.gcd(num_features, num_groups)
        norm_layer = nn.GroupNorm(num_channels=num_features, num_groups=num_groups)
    elif norm_type == 'instance_norm_2d':
        norm_layer = nn.InstanceNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == 'instance_norm_1d':
        norm_layer = nn.InstanceNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type == 'layer_norm':
        norm_layer = nn.LayerNorm(num_features)
    elif norm_type == 'identity':
        norm_layer = Identity()
    else:
        raise ValueError(
            'Supported normalization layer arguments are: {}. Got: {}'.format(SUPPORTED_NORM_FNS, norm_type))
    return norm_layer

SUPPORTED_ACT_FNS = [
    'relu', 
    'prelu', 
    'relu6', 
    'leaky_relu', 
    'gelu',
    'sigmoid', 
    'hard_sigmoid', 
    'swish', 
    'hard_swish'
]

def get_activation_fn(act_type = 'swish', 
                      num_parameters = -1, 
                      inplace = True,
                      negative_slope = 0.1
                      ):
    if act_type == 'relu':
        return nn.ReLU(inplace=False)
    elif act_type == 'prelu':
        assert num_parameters >= 1
        return nn.PReLU(num_parameters=num_parameters)
    elif act_type == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif act_type == 'hard_sigmoid':
        return nn.Hardsigmoid(inplace=inplace)
    elif act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'sigmoid':
        return nn.Sigmoid()
    elif act_type == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_type == 'hard_swish':
        return nn.Hardswish(inplace=inplace)
    else:
        raise ValueError(
            'Supported activation layers are: {}. Supplied argument is: {}'.format(SUPPORTED_ACT_FNS, act_type))

class ConvBlock(nn.Module):
    """
        2D Convolution block with normalization and activation layer
    """
    def __init__(self, 
                 opts, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride = 1,
                 dilation = 1, 
                 groups = 1,
                 bias = False, 
                 padding_mode = 'zeros',
                 use_norm = True, 
                 use_act = True
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param kernel_size: kernel size
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.
            :param groups: number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
            :param bias: use bias or not
            :param padding_mode: padding mode
            :param use_norm: use normalization layer after convolution layer or not
            :param use_act: Use activation layer or not
        """
        super(ConvBlock, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        if in_channels % groups != 0:
            raise ValueError('Input channels are not divisible by groups. {}%{} != 0 '.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('Output channels are not divisible by groups. {}%{} != 0 '.format(out_channels, groups))

        block = nn.Sequential()

        conv_layer = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size,
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation, 
                               groups=groups, bias=bias,
                               padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "prelu")

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(act_type=act_type,
                                          inplace=inplace,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x):
        return self.block(x)

mobilevit_xxs_cfg = {
    "layer1": {
        "out_channels": 16,
        "expand_ratio": 2,
        "num_blocks": 1,
        "stride": 1,
        "block_type": "mv2"
    },
    "layer2": {
        "out_channels": 24,
        "expand_ratio": 2,
        "num_blocks": 3,
        "stride": 2,
        "block_type": "mv2"
    },
    "layer3": {  # 28x28
        "out_channels": 48,
        "transformer_channels": 64,
        "ffn_dim": 128,
        "transformer_blocks": 2,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer4": {  # 14x14
        "out_channels": 64,
        "transformer_channels": 80,
        "ffn_dim": 160,
        "transformer_blocks": 4,
        "patch_h": 2, 
        "patch_w": 2, 
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "layer5": {  # 7x7
        "out_channels": 80,
        "transformer_channels": 96,
        "ffn_dim": 192,
        "transformer_blocks": 3,
        "patch_h": 2,
        "patch_w": 2,
        "stride": 2,
        "mv_expand_ratio": 2,
        "num_heads": 4,
        "block_type": "mobilevit"
    },
    "last_layer_exp_factor": 4
}    

model_cfg = {
    'mobilevit_xxs': mobilevit_xxs_cfg, 
    }

def get_config(opts):
    model_name = getattr(opts, "model_name", 'mobilevit_xxs')
    return model_cfg[model_name]

pool_types = ['mean', 'rms', 'abs']

class GlobalPool(nn.Module):
    """
        Global pooling 
    """
    def __init__(self, pool_type='mean', keep_dim=False):
        """
            :param pool_type: Global pool operation type (mean, rms, abs)
            :param keep_dim: Keep dimensions the same as the input or not
        """
        super(GlobalPool, self).__init__()
        if pool_type not in pool_types:
            raise ValueError('Supported pool types are: {}. Got {}'.format(pool_types, pool_type))
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x):
        assert x.dim() == 4, "Got: {}".format(x.shape)
        if self.pool_type == 'rms':
            x = x ** 2
            x = torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
            x = x ** -0.5
        elif self.pool_type == 'abs':
            x = torch.mean(torch.abs(x), dim=[-2, -1], keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
        return x

    def forward(self, x):
        return self._global_pool(x)

supported_conv_inits = [
    'kaiming_normal', 
    'kaiming_uniform', 
    'xavier_normal', 
    'xavier_uniform', 
    'normal', 
    'trunc_normal'
    ]

def _init_nn_layers(module, init_method = 'kaiming_normal', std_val = None):
    init_method = init_method.lower()
    if init_method == 'kaiming_normal':
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'kaiming_uniform':
        if module.weight is not None:
            nn.init.kaiming_uniform_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'xavier_normal':
        if module.weight is not None:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'xavier_uniform':
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'normal':
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) if std_val is None else std_val
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'trunc_normal':
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) if std_val is None else std_val
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    else:
        supported_conv_message = 'Supported initialization methods are:'
        for i, l in enumerate(supported_conv_inits):
            supported_conv_message += '\n \t {}) {}'.format(i, l)
        raise ValueError('{} \n Got: {}'.format(supported_conv_message, init_method))
    

def initialize_conv_layer(module, init_method = 'kaiming_normal', std_val = 0.01):
    _init_nn_layers(module=module, init_method=init_method, std_val=std_val)

def initialize_fc_layer(module, init_method = 'normal', std_val = 0.01):
    if hasattr(module, "layer"):
        _init_nn_layers(module=module.layer, init_method=init_method, std_val=std_val)
    else:
        _init_nn_layers(module=module, init_method=init_method, std_val=std_val)

def initialize_norm_layers(module):
    def _init_fn(module):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    _init_fn(module.layer) if hasattr(module, "layer") else _init_fn(module=module)

norm_layers_tuple = (
    nn.BatchNorm1d, 
    nn.BatchNorm2d, 
    nn.SyncBatchNorm, 
    nn.LayerNorm, 
    nn.InstanceNorm1d, 
    nn.InstanceNorm2d, 
    nn.GroupNorm
    )

def initialize_weights(opts, modules):
    # weight initialization
    conv_init_type = getattr(opts, "model.layer.conv_init", "kaiming_normal")
    linear_init_type = getattr(opts, "model.layer.linear_init", "normal")

    conv_std = getattr(opts, "model.layer.conv_init_std_dev", None)
    linear_std = getattr(opts, "model.layer.linear_init_std_dev", 0.01)
    group_linear_std = getattr(opts, "model.layer.group_linear_init_std_dev", 0.01)

    for m in modules:
        if isinstance(m, nn.Conv2d):
            initialize_conv_layer(module=m, init_method=conv_init_type, std_val=conv_std)
        elif isinstance(m, norm_layers_tuple):
            initialize_norm_layers(module=m)
        elif isinstance(m, nn.Linear):
            initialize_fc_layer(module=m, init_method=linear_init_type, std_val=linear_std)

from typing import Union, Optional
def make_divisible(v: Union[float, int],
                   divisor: Optional[int] = 8,
                   min_value: Optional[Union[float, int]] = None
                   ):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    """
        Inverted residual block (MobileNetv2)
    """
    def __init__(self,
                 opts,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation = 1
                 ):
        """
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: move the kernel by this amount during convolution operation
            :param expand_ratio: expand ratio for hidden dimension
            :param dilation: add zeros between kernel elements to increase the effective receptive field of the kernel.
        """
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvBlock(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=True, use_norm=True))

        block.add_module(
            name="conv_3x3",
            module=ConvBlock(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                             groups=hidden_dim, use_act=True, use_norm=True, dilation=dilation)
        )

        block.add_module(name="red_1x1",
                         module=ConvBlock(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class MobileViT(nn.Module):
    """
        MobileViT: https://arxiv.org/pdf/2110.02178
    """
    def __init__(self, opts):
        image_channels, input_channels = 3, 16
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.1)

        # original mobilevit uses swish activation function
        setattr(opts, "model.activation.name", "swish")

        mobilevit_config = get_config(opts=opts)

        super(MobileViT, self).__init__()

        self.dilation = 1
        self.conv_1 = ConvBlock(
                opts=opts, in_channels=image_channels, out_channels=input_channels,
                kernel_size=3, stride=2, use_norm=True, use_act=True
            )

        self.layer_1, self.layer_1_channels = self._make_layer(
            opts=opts, input_channel=input_channels, cfg=mobilevit_config["layer1"]
        )

        self.layer_2, self.layer_2_channels = self._make_layer(
            opts=opts, input_channel=self.layer_1_channels, cfg=mobilevit_config["layer2"]
        )

        self.layer_3, self.layer_3_channels = self._make_layer(
            opts=opts, input_channel=self.layer_2_channels, cfg=mobilevit_config["layer3"]
        )

        self.layer_4, self.layer_4_channels = self._make_layer(
            opts=opts, input_channel=self.layer_3_channels, cfg=mobilevit_config["layer4"],
        )

        self.layer_5, self.layer_5_channels = self._make_layer(
            opts=opts, input_channel=self.layer_4_channels, cfg=mobilevit_config["layer5"], 
        )

        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * self.layer_5_channels, 960)
        self.conv_1x1_exp = ConvBlock(
                opts=opts, in_channels=self.layer_5_channels, out_channels=exp_channels,
                kernel_size=1, stride=1, use_act=True, use_norm=True
            )

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool())
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=classifier_dropout, inplace=True))
        self.classifier.add_module(
            name="fc",
            module=nn.Linear(in_features=exp_channels, out_features=num_classes, bias=True)
        )

        # weight initialization
        self.reset_parameters(opts=opts)

    def reset_parameters(self, opts):
        initialize_weights(opts=opts, modules=self.modules())

    def extract_features(self, x):
        out_dict = {} # consider input size of 224
        x = self.conv_1(x) # 112 x112
        x = self.layer_1(x) # 112 x112
        out_dict["out_l1"] = x  # level-1 feature

        x = self.layer_2(x) # 56 x 56
        out_dict["out_l2"] = x

        x = self.layer_3(x) # 28 x 28
        out_dict["out_l3"] = x

        x = self.layer_4(x) # 14 x 14
        out_dict["out_l4"] = x

        x = self.layer_5(x) # 7 x 7
        out_dict["out_l5"] = x

        if self.conv_1x1_exp is not None:
            x = self.conv_1x1_exp(x) # 7 x 7
            out_dict["out_l5_exp"] = x

        return out_dict, x

    def forward(self, x):
        _, x = self.extract_features(x)
        x = self.classifier(x)
        return x

    def _make_layer(self, opts, input_channel, cfg, dilate = False):
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel, cfg):
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            input_channel = output_channels
            block.append(layer)
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg, dilate = False):
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        block.append(
            MobileViTBlock(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.classification.mit.attn_dropout", 0.0),
                head_dim=head_dim,
                conv_ksize=getattr(opts, "model.classification.mit.conv_kernel_size", 3)
            )
        )

        return nn.Sequential(*block), input_channel
    

def mobilevit_xxs(num_classes=1000, pretrained=False, device='cpu'):
    model_name = 'mobilevit_xxs'
    opts = Namespace(model_name = model_name)
    model = MobileViT(opts=opts)
    if False and pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt', map_location=torch.device(device)
        )
        model.load_state_dict(ckpt)

    # Initialize new head only of classes != 1000.
    if num_classes != 1000:
        print('Initializing new head')
        in_features = model.classifier.fc.in_features
        model.classifier.fc = nn.Linear(
            in_features=in_features, out_features=num_classes, bias=True
        )
    return model


def create_model(num_classes, pretrained=True, coco_model=False, device='cpu'):
    # Load the backbone.
    model_backbone = mobilevit_xxs(100, pretrained=pretrained, device='cpu')

    backbone = nn.Sequential(*list(model_backbone.children())[:-1])

    # Output channels from the final convolutional layer.
    backbone.out_channels = 320

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
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

np.random.seed(2023)
ckpt = torch.load(PATH_TO_WEIGHTS, map_location=args["device"])
NUM_CLASSES = ckpt['data']['NC']
CLASSES = ckpt['data']['CLASSES']
model = create_model(
        NUM_CLASSES, True, coco_model=False, device=args["device"]
    )
ckpt_state_dict = ckpt['model_state_dict']
model.load_state_dict(ckpt_state_dict)

model.eval()


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
                ##frame = jpg.copy()
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
     global LAST_CAP_FRAME, CAMERA_BEGIN_TIME, CAMERA_TIME_LIMIT_SECONDS
     wtfevloop = asyncio.get_event_loop()
     ##frame_prev = None
     while(True):
        if(time.time()-CAMERA_BEGIN_TIME +3 >CAMERA_TIME_LIMIT_SECONDS): return
        if(LAST_CAP_FRAME is None):
            await asyncio.sleep(1)
            continue
       ## if(frame_prev == LAST_CAP_FRAME):
       ##     await asyncio.sleep(1)
       ##     continue
       ## frame_prev = LAST_CAP_FRAME
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