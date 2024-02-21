#adb forward tcp:8080 tcp:8080
import datetime
MODEL_NAME_SUFFIX_IMG_NAMING="_detectron_02-20"
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
PATH_TO_WEIGHTS = "bin/2024-02-20/fb_detectron/model_0099999.pth"
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
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
#cap, frame_width, frame_height = read_return_video_data(VIDEO_PATH)
#args ={"classes": None,
#       "track": False}
CLASSES_BEAUTY=["background", "Cz"]
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
SHOW_PREVIEW = True#
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
runner = setup_after_launch(cfg, output_dir, runner_class)
checkpointer = runner.build_checkpointer(cfg, model, save_dir=output_dir)
if resume and checkpointer.has_checkpoint():
    checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
else:
    checkpoint = checkpointer.load(cfg.MODEL.WEIGHTS)
train_iter = checkpoint.get("iteration", None)
model.eval()
#model.eval()
metrics = runner.do_test(cfg, model, train_iter=train_iter)

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

def recognize(jpg):
    frame_count += 1
"""
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
"""

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