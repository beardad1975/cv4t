import time
import tempfile
import math
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
        #if len(detection_result.hand_landmarks) >= 1:
        if detection_result.hand_landmarks:
            self.last_result = detection_result
        else:
            self.last_result = None

global_callback = Callback()
    
    
#### bundle tflite every time in temp dir 

def bundle_and_load_hand_model():
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

    ### add gesture model
    temp_final_output_path = Path(temp_folder.name) / 'gesture_recognizer.task'

    input_models = {}
    gesture_path = Path(__file__).parent / 'model_needed' / 'hand_gesture_recognizer.task' 
    with open(gesture_path, 'rb') as f:
        gesture_model = f.read()
    input_models['hand_gesture_recognizer.task'] = gesture_model

    with open(temp_output_path, 'rb') as f:
        temp_output_model = f.read()
    input_models['hand_landmarker.task'] = temp_output_model

    model_asset_bundle_utils.create_model_asset_bundle(
        input_models, temp_final_output_path
    )

    # load model
    with open(temp_final_output_path, 'rb') as f:
        model = f.read()    

    return model

## ori version
# def bundle_and_load_hand_model():
#     temp_folder = tempfile.TemporaryDirectory()
#     temp_output_path = Path(temp_folder.name) / 'hand_landmarker.task'

#     input_models = {}

#     hand_landmark_path = Path(mp.__path__[0]) / 'modules'/ 'hand_landmark' / 'hand_landmark_full.tflite' 
#     with open(hand_landmark_path, 'rb') as f:
#         hand_landmark_model = f.read()
#     input_models['hand_landmarks_detector.tflite'] = hand_landmark_model

#     palm_path = Path(mp.__path__[0]) / 'modules'/ 'palm_detection' / 'palm_detection_full.tflite' 
#     with open(palm_path, 'rb') as f:
#         palm_model = f.read()
#     input_models['hand_detector.tflite'] = palm_model

#     model_asset_bundle_utils.create_model_asset_bundle(
#         input_models, temp_output_path
#     )

#     # load model
#     with open(temp_output_path, 'rb') as f:
#         model = f.read()

#     return model

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
        self.mp_detector.recognize_async(rgb_frame, timestamp.microseconds())
        #self.mp_detector.detect_async(rgb_frame, timestamp.microseconds())
        return HandDetectionResult(img_width, img_height)

class HandDetectionResult():
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __bool__(self):

        return bool(global_callback.last_result)

    def __len__(self):
        result = global_callback.last_result

        if not result.hand_landmarks:
            print('沒有偵測到手部')
            return 0

        
        return len(result.hand_landmarks)
    

class HandInfo():
    def __init__(self, idx, img_width, img_height):
        self.idx = idx
        self.img_width = img_width
        self.img_height = img_height

    def 特徵點(self, 索引):
        result = global_callback.last_result
        try:
            x = math.floor(result.hand_landmarks[self.idx][索引].x * self.img_width)
            y = math.floor(result.hand_landmarks[self.idx][索引].y * self.img_height)
        except (AttributeError, IndexError):
            print('info: 取不到特徵資料,xy傳回0')
            x = 0
            y = 0
        return (x, y)
    
    def x(self, 索引):
        result = global_callback.last_result
        try:
            x = math.floor(result.hand_landmarks[self.idx][索引].x * self.img_width)
        except (AttributeError, IndexError):
            print('info: 取不到特徵資料,x傳回0')
            x = 0
        return x
    
    def y(self, 索引):
        result = global_callback.last_result
        try:
            y = math.floor(result.hand_landmarks[self.idx][索引].y * self.img_height)
        except (AttributeError, IndexError):
            print('info: 取不到特徵資料,y傳回0')
            y = 0    
        return y
    
    def z(self, 索引):
        result = global_callback.last_result
        try:
            z = math.floor(result.hand_landmarks[self.idx][索引].z * self.img_width)
        except (AttributeError, IndexError):
            print('info: 取不到特徵資料,z傳回0')
            z = 0 
        return z

    def 座標3D(self, 索引):
        result = global_callback.last_result
        try:
            x = math.floor(result.hand_landmarks[self.idx][索引].x * self.img_width)
            y = math.floor(result.hand_landmarks[self.idx][索引].y * self.img_height)
            z = math.floor(result.hand_landmarks[self.idx][索引].z * self.img_width)
        except (AttributeError, IndexError):
            print('info: 取不到特徵資料,xyz傳回0')
            x, y, z = 0, 0, 0 
        return (x, y, z)

    @property
    def handedness(self):
        result = global_callback.last_result
        try:
            return result.handedness[self.idx][0].category_name
        except (AttributeError, IndexError):
            print('info: 取不到特徵資料,傳回空字串')
            return ""

    @property
    def gesture(self):
        result = global_callback.last_result
        try:
            return result.gestures[self.idx][0].category_name 
        except (AttributeError, IndexError):
            print('info: 取不到特徵資料,傳回None')
            return "None"   


### main function

def 設置HandAndGesture(手數量=2):
#def 設置HandDetection(手數量=2):
    if 手數量 > 5:
        手數量 = 5
    model = bundle_and_load_hand_model()
    base_options = python.BaseOptions(model_asset_buffer=model)
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.GestureRecognizerOptions(base_options=base_options,
                                        running_mode=VisionRunningMode.LIVE_STREAM,
                                        result_callback=global_callback.check_result,
                                        num_hands=手數量)
    # options = vision.HandLandmarkerOptions(base_options=base_options,
    #                                     running_mode=VisionRunningMode.LIVE_STREAM,
    #                                     result_callback=global_callback.check_result,
    #                                     num_hands=手數量)
    mp_detector = vision.GestureRecognizer.create_from_options(options)
    # mp_detector = vision.HandLandmarker.create_from_options(options)
    return HandDetectorWrap(mp_detector)

def 標記Hand(img, result_wrap):
    if not result_wrap:
        print('info: 沒有偵測到手,不標示')
        return
    #print('mark here')
    draw_landmarks_on_image(img, global_callback.last_result)



def 取出Hand(result_wrap):
    if not result_wrap:
        print('info: 沒有偵測到手,無資料')
        return
    
    result = global_callback.last_result
    if len(result.hand_landmarks) == 0 :
        raise IndentationError

    return HandInfo(0, result_wrap.img_width, result_wrap.img_height)

def 取出Hand清單(result_wrap):
    if not result_wrap:
        #print('info: 沒有偵測到手,無資料')
        return []

    result = global_callback.last_result
    hand_list = []
    for idx, _ in enumerate(result.hand_landmarks):
        hand_list.append(HandInfo(idx, result_wrap.img_width, result_wrap.img_height))

    return hand_list