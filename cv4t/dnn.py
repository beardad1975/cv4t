import numpy as np
import cv2
from pathlib import Path


models_path = Path(__file__).parent / 'models'
face_prototxt_path = str(models_path / 'face_deploy.prototxt')
face_caffemodel_path = str(models_path / 'face_res10_300x300_ssd_iter_140000.caffemodel')
landmark_prototxt_path = str(models_path / 'landmark_deploy.prototxt')
landmark_caffemodel_path = str(models_path / 'landmark_vanface.caffemodel')

face_net = None
landmark_net = None

def 深度學習人臉模型():
    global face_net, landmark_net
    if face_net is None:
        face_net = cv2.dnn.readNetFromCaffe(face_prototxt_path, face_caffemodel_path)

    if landmark_net is None:
        landmark_net = cv2.dnn.readNetFromCaffe(landmark_prototxt_path, landmark_caffemodel_path)
        
    return NeuralNetwork(face_net, landmark_net)

class NeuralNetwork:
    def __init__(self, face_net, landmark_net, min_confidence=0.6):
        self.face_net = face_net
        self.landmark_net = landmark_net
        self.min_confidence = min_confidence

    def 設定輸入(self, img):
        self.img = img
        self.height, self.width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)

    def 正向傳播(self):
        face_list = []
        detections = self.face_net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.min_confidence:
                continue
            multiple_array = np.array([self.width, self.height, self.width, self.height])
            box = detections[0, 0, i, 3:7] * multiple_array
            
            x1, y1, x2, y2 = box.astype("int")
            if not 0 <= x1 < self.width: continue
            if not 0 <= x2 < self.width: continue
            if not 0 <= y1 < self.height: continue
            if not 0 <= y2 < self.height: continue

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

        

    def 偵測特徵點(self):
        LM_caffe_param = 60
        
        
        l,t,r,b = self.x1, self.y1, self.x2, self.y2
        roi = self.nn.img[t:b+1, l:r+1]
        #print(l,t,r,b)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray_roi, (LM_caffe_param, LM_caffe_param)).astype(np.float32)
        
        m = np.zeros((LM_caffe_param,LM_caffe_param))
        sd = np.zeros((LM_caffe_param,LM_caffe_param))
        mean, std_dev = cv2.meanStdDev(res, m, sd)
        normalized_roi = (res - mean[0][0]) / (0.000001 + std_dev[0][0])

        blob = cv2.dnn.blobFromImage(normalized_roi, 1.0,
        (LM_caffe_param, LM_caffe_param), None)
        self.nn.landmark_net.setInput(blob)
        caffe_landmark = self.nn.landmark_net.forward()
        

        
        for landmark in caffe_landmark:
            # only for 1st landmark
            point_list = []
            for i in range(len(landmark)//2):
                x = landmark[2*i] * (r-l) + l
                y = landmark[2*i+1] * (b-t) + t
                #x = x if x >= 0 else 0
                #y = y if y >= 0 else 0
                point_list.append((int(x),int(y)))
            break

        return LM68List(point_list)

    @property   
    def 矩形左上點(self):
        return (self.x1, self.y1)

    @property   
    def 矩形左下點(self):
        return (self.x1, self.y2)

    @property   
    def 矩形右上點(self):
        return (self.x2, self.y1)  

    @property   
    def 矩形右下點(self):
        return (self.x2, self.y2)

    @property   
    def 矩形中心點(self):
        return (self.x2+self.x1)//2, (self.y2+self.y1)//2  

    @property   
    def 信心值(self):
        return self.confidence

    @property
    def 陣列(self):
        return self.nn.img[self.y1:self.y2+1, self.x1:self.x2+1]

    @property
    def 寬度(self):
        return self.x2 - self.x1    

    @property
    def 高度(self):
        return self.y2 - self.y1 

class LM68List:
    def __init__(self, point_list):
        self.point_list = point_list

    def __iter__(self):
        return iter(self.point_list) 

    @property
    def 全部清單(self):
        return self.point_list

    @property
    def 下巴清單(self):
        return self.point_list[:17] 

    @property
    def 眉毛清單(self):
        return self.point_list[17:27]    

    @property
    def 左眉毛清單(self):
        return self.point_list[17:22] 

    @property
    def 右眉毛清單(self):
        return self.point_list[22:27]

    @property
    def 鼻清單(self):
        return self.point_list[27:36]

    @property
    def 鼻梁清單(self):
        return self.point_list[27:31]    

    @property
    def 鼻翼清單(self):
        return self.point_list[31:36]  

    @property
    def 眼清單(self):
        return self.point_list[36:48]  

    @property
    def 左眼清單(self):
        return self.point_list[36:42]   

    @property
    def 右眼清單(self):
        return self.point_list[42:48]    

    @property
    def 脣清單(self):
        return self.point_list[48:60]

    @property
    def 口清單(self):
        return self.point_list[60:68]