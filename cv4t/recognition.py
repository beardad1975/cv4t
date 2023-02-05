import mediapipe as mp
import cv2 
import math

mp_drawing = mp.solutions.drawing_utils

class FaceDetectorWrap():
    def __init__(self, mp_detector):
        self.mp_detector = mp_detector

    def process(self, img):
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_result = self.mp_detector.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return FaceResultWrap(mp_result)

class FaceResultWrap():
    def __init__(self, mp_result):
        self.mp_result = mp_result

    def __bool__(self):
        return bool(self.mp_result.detections)


def 產生FaceDetection(模型選擇=1, 最小信心值=0.5):
    mp_detector =  mp.solutions.face_detection.FaceDetection(model_selection=模型選擇,
                                min_detection_confidence=最小信心值)
    return FaceDetectorWrap(mp_detector)

def 標示Face(img, result_wrap):
    if not result_wrap.mp_result.detections:
        print('無偵測資訊,不標示')
        return
      
    for detection in result_wrap.mp_result.detections:
        mp_drawing.draw_detection(img, detection)
        

    

