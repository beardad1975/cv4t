import time
import tempfile
from pathlib import Path

import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.metadata.metadata_writers import model_asset_bundle_utils


### Visualization utilities , from mediapipe example
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    #annotated_image = np.copy(rgb_image)
    annotated_image = rgb_image

  # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

    # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    return

### Result call back

class Callback:
    def __init__(self):
        self.last_result = None

    def check_result(self, detection_result, input_image,timestamp_ms):
        if detection_result.hand_landmarks:
            self.last_result = detection_result
        else:
            self.last_result = None

global_callback = Callback()
    
    
#### bundle tflite every time in temp dir 

def bundle_and_load_model():
    temp_folder = tempfile.TemporaryDirectory()
    temp_output_path = Path(temp_folder.name) / 'hand_landmarker.task'

    input_models = {}

    hand_landmark_path = Path(mp.__path__[0]) / 'modules'/ 'hand_landmark' / 'hand_landmark_full.tflite' 
    with open(hand_landmark_path, 'rb') as f:
        hand_landmark_model = f.read()
    input_models['hand_landmarks_detector.tflite'] = hand_landmark_model

    palm_path = Path(mp.__path__[0]) / 'modules'/ 'palm_detection' / 'palm_detection_full.tflite' 
    with open(palm_path, 'rb') as f:
        palm_model = f.read()
    input_models['hand_detector.tflite'] = palm_model

    model_asset_bundle_utils.create_model_asset_bundle(
        input_models, temp_output_path
    )

    # load model
    with open(temp_output_path, 'rb') as f:
        model = f.read()

    return model

### custum detector and result class

class HandDetectorWrap():
    def __init__(self, mp_detector):
        self.mp_detector = mp_detector

    def process(self, img):
        # prepare
        img_height, img_width, _ = img.shape
        cv_mat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
        timestamp = mp.Timestamp.from_seconds(time.time())
        # detect for live stream
        self.mp_detector.detect_async(rgb_frame, timestamp.microseconds())
        return HandDetectionResult(img_width, img_height)

class HandDetectionResult():
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __bool__(self):

        return bool(global_callback.last_result)

    # def __len__(self):
    #     if self.mp_result.pose_landmarks:
    #         return 1
    #     else:
    #         print('沒有偵測到姿勢,無資料')
    #         return 0



### main function

def 設置HandDetection():
    model = bundle_and_load_model()
    base_options = python.BaseOptions(model_asset_buffer=model)
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        running_mode=VisionRunningMode.LIVE_STREAM,
                                        result_callback=global_callback.check_result,
                                        num_hands=2)
    mp_detector = vision.HandLandmarker.create_from_options(options)
    return HandDetectorWrap(mp_detector)

def 標記Hand(img, result_wrap):
    if not result_wrap:
        print('info: 沒有偵測到手,不標示')
        return
    #print('mark here')
    draw_landmarks_on_image(img, global_callback.last_result)
