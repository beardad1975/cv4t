import numpy as np
import cv2
from pathlib import Path


models_path = Path(__file__).parent / 'models'
face_prototxt_path = str(models_path / 'face_deploy.prototxt')
face_caffemodel_path = str(models_path / 'face_res10_300x300_ssd_iter_140000.caffemodel')
landmark_prototxt_path = str(models_path / 'landmark_deploy.prototxt')
landmark_caffemodel_path = str(models_path / 'landmark_vanface.caffemodel')

face_nn = None
landmark_nn = None

def 深度學習人臉模型():
    global face_nn
    if face_nn is None:
        face_nn = cv2.dnn.readNetFromCaffe(face_prototxt_path, face_caffemodel_path)
        return NeuralNetwork(face_nn)

class NeuralNetwork:
    def __init__(self, nn, min_confidence=0.6):
        self.nn = nn
        self.min_confidence = min_confidence

    def 設定輸入(self, img):
        self.img = img
        self.height, self.width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.nn.setInput(blob)

    def 正向傳播(self):
        face_list = []
        detections = self.nn.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.min_confidence:
                continue
            multiple_array = np.array([self.width, self.height, self.width, self.height])
            box = detections[0, 0, i, 3:7] * multiple_array
            
            x1, y1, x2, y2 = box.astype("int")
            face_list.append(Face(self, x1, y1, x2, y2, confidence))        
        return  face_list

class Face:
    def __init__(self, nn, x1, y1, x2, y2, confidence):
        self.nn = nn
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence 

    @property   
    def 左上點(self):
        return (self.x1, self.y1)

    @property   
    def 右下點(self):
        return (self.x2, self.y2)    

    @property   
    def 信心值(self):
        return self.confidence



