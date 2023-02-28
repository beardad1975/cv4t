import mediapipe as mp
import cv2 
import math

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

## Face Detection ==============================

class FaceDetectorWrap():
    def __init__(self, mp_detector):
        self.mp_detector = mp_detector

    def process(self, img):
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_result = self.mp_detector.process(img_rgb)
        img.flags.writeable = True
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_height, img_width, _ = img.shape
        return FaceDetectionResult(mp_result, img_width, img_height)

class FaceDetectionResult():
    def __init__(self, mp_result, img_width, img_height):
        self.mp_result = mp_result
        self.img_width = img_width
        self.img_height = img_height

    def __bool__(self):
        return bool(self.mp_result.detections)

    def __len__(self):
        if self.mp_result.detections:
            return len(self.mp_result.detections)
        else:
            #print('沒有偵測到人臉,無資料')
            return 0

class FaceInfo():
    def __init__(self, detection, result_wrap):
        self.mp_detection = detection
        self.img_width = result_wrap.img_width
        self.img_height = result_wrap.img_height

    @property
    def score(self):
        return round(self.mp_detection.score[0], 2)

    @property
    def upper_left(self):
        bbox_xmin = self.mp_detection.location_data.relative_bounding_box.xmin
        x = math.floor(bbox_xmin * self.img_width)
        bbox_ymin = self.mp_detection.location_data.relative_bounding_box.ymin
        y = math.floor(bbox_ymin * self.img_height)
        return (x, y)

    @property
    def bottom_right(self):
        bbox_xmin = self.mp_detection.location_data.relative_bounding_box.xmin
        bbox_ymin = self.mp_detection.location_data.relative_bounding_box.ymin
        bbox_width = self.mp_detection.location_data.relative_bounding_box.width
        bbox_height = self.mp_detection.location_data.relative_bounding_box.height

        x = math.floor((bbox_xmin+bbox_width) * self.img_width)
        y = math.floor((bbox_ymin+bbox_height) * self.img_height)
        return (x, y)
    
    @property
    def upper_right(self):
        bbox_xmin = self.mp_detection.location_data.relative_bounding_box.xmin
        bbox_ymin = self.mp_detection.location_data.relative_bounding_box.ymin
        bbox_width = self.mp_detection.location_data.relative_bounding_box.width
        

        x = math.floor((bbox_xmin+bbox_width) * self.img_width)
        y = math.floor(bbox_ymin * self.img_height)
        return (x, y)

    @property
    def bottom_left(self):
        bbox_xmin = self.mp_detection.location_data.relative_bounding_box.xmin
        bbox_ymin = self.mp_detection.location_data.relative_bounding_box.ymin
       
        bbox_height = self.mp_detection.location_data.relative_bounding_box.height

        x = math.floor(bbox_xmin * self.img_width)
        y = math.floor((bbox_ymin+bbox_height) * self.img_height)
        return (x, y)

    @property
    def right_eye(self):
        keypoints = self.mp_detection.location_data.relative_keypoints
        x = math.floor(keypoints[0].x * self.img_width)
        y = math.floor(keypoints[0].y * self.img_height)
        return (x, y)

    @property
    def left_eye(self):
        keypoints = self.mp_detection.location_data.relative_keypoints
        x = math.floor(keypoints[1].x * self.img_width)
        y = math.floor(keypoints[1].y * self.img_height)
        return (x, y)

    @property
    def nose_tip(self):
        keypoints = self.mp_detection.location_data.relative_keypoints
        x = math.floor(keypoints[2].x * self.img_width)
        y = math.floor(keypoints[2].y * self.img_height)
        return (x, y)

    @property
    def mouth_center(self):
        keypoints = self.mp_detection.location_data.relative_keypoints
        x = math.floor(keypoints[3].x * self.img_width)
        y = math.floor(keypoints[3].y * self.img_height)
        return (x, y)

    @property
    def right_ear_tragion(self):
        keypoints = self.mp_detection.location_data.relative_keypoints
        x = math.floor(keypoints[4].x * self.img_width)
        y = math.floor(keypoints[4].y * self.img_height)
        return (x, y)

    @property
    def left_ear_tragion(self):
        keypoints = self.mp_detection.location_data.relative_keypoints
        x = math.floor(keypoints[5].x * self.img_width)
        y = math.floor(keypoints[5].y * self.img_height)
        return (x, y)


# 最小信心值 效果試不出來
def 設置FaceDetection(模型選擇=1, 最小偵測信心=0.5):
    mp_detector =  mp.solutions.face_detection.FaceDetection(model_selection=模型選擇,
                                min_detection_confidence=最小偵測信心)
    return FaceDetectorWrap(mp_detector)

def 標記Face(img, result_wrap):
    # 標記 矩形關鍵點
    if not result_wrap:
        print('info: 沒有偵測到人臉,不標示')
        return

     
    for detection in result_wrap.mp_result.detections:
        mp_drawing.draw_detection(img, detection)

def 取出Face(result_wrap):
    if not result_wrap:
        print('info: 沒有偵測到人臉,無資料')
        return
    
    detection = result_wrap.mp_result.detections[0]
    return FaceInfo(detection, result_wrap)

def 取出Face清單(result_wrap):
    if not result_wrap:
        print('info: 沒有偵測到人臉,無資料')
        return []

    face_list = [FaceInfo(d, result_wrap) for d in result_wrap.mp_result.detections ]
    
    return face_list

## FaceMesh  ================================

class FaceMeshWrap():
    def __init__(self, mp_detector):
        self.mp_detector = mp_detector

    def process(self, img):
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_result = self.mp_detector.process(img_rgb)
        img.flags.writeable = True
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_height, img_width, _ = img.shape
        return FaceMeshResult(mp_result, img_width, img_height)

class FaceMeshResult():
    def __init__(self, mp_result, img_width, img_height):
        self.mp_result = mp_result
        self.img_width = img_width
        self.img_height = img_height

    def __bool__(self):
        return bool(self.mp_result.multi_face_landmarks)

    def __len__(self):
        if self.mp_result.multi_face_landmarks:
            return len(self.mp_result.multi_face_landmarks)
        else:
            #print('沒有偵測到人臉,無資料')
            return 0

class FaceLandmarksInfo():
    def __init__(self, face_landmarks, result_wrap):
        self.mp_face_landmarks = face_landmarks
        self.img_width = result_wrap.img_width
        self.img_height = result_wrap.img_height

    def __getitem__(self, item):
        item = self.mp_face_landmarks.landmark[item]

        return (math.floor(item.x * self.img_width),
                 math.floor(item.y * self.img_height))

    def __len__(self):
        if self.mp_face_landmarks.landmark:
            return len(self.mp_face_landmarks.landmark)
        else:
            return 0

class Face3DLandmarksInfo():
    def __init__(self, face_landmarks, result_wrap):
        self.mp_face_landmarks = face_landmarks
        self.img_width = result_wrap.img_width
        self.img_height = result_wrap.img_height

    def __getitem__(self, item):
        item = self.mp_face_landmarks.landmark[item]

        return (math.floor(item.x * self.img_width),
                 math.floor(item.y * self.img_height),
                 math.floor(item.z * self.img_width))

    def __len__(self):
        if self.mp_face_landmarks.landmark:
            return len(self.mp_face_landmarks.landmark)
        else:
            return 0

    # @property
    # def 信心值(self):
    #     return round(self.mp_detection.score[0], 2)

def 設置FaceMesh(靜態影像模式=False, 最大偵測數=1, 精細特徵點=True,  最小偵測信心=0.5, 最小追蹤信心=0.5):
    mp_detector =  mp.solutions.face_mesh.FaceMesh(
        static_image_mode=靜態影像模式,
        max_num_faces=最大偵測數,
        refine_landmarks=精細特徵點,
        min_detection_confidence=最小偵測信心,
        min_tracking_confidence=最小追蹤信心)
    return FaceMeshWrap(mp_detector)

def 取出Landmarks(result_wrap):
    if not result_wrap:
        print('info: 沒有偵測到人臉,無資料')
        return
    
    face_landmarks = result_wrap.mp_result.multi_face_landmarks[0]
    return FaceLandmarksInfo(face_landmarks, result_wrap)

def 取出3DLandmarks(result_wrap):
    if not result_wrap:
        print('info: 沒有偵測到人臉,無資料')
        return
    
    face_landmarks = result_wrap.mp_result.multi_face_landmarks[0]
    return Face3DLandmarksInfo(face_landmarks, result_wrap)

def 標記FaceMesh(img, result_wrap, type='FACE_MESH'):
    if not result_wrap:
        print('info: 沒有偵測到人臉,無法標示')
        return

    if type == 'FACE_MESH':
        drawing_spec = mp_drawing.DrawingSpec(color=(180,180,180),thickness=1)
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_TESSELATION
        connection_drawing_spec = drawing_spec
    elif type == 'FACE_LANDMARKS':
        drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0),thickness=-1, circle_radius=1)
        landmark_drawing_spec = drawing_spec
        connections = None
        connection_drawing_spec = None    
    elif type == 'CONTOURS':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_CONTOURS
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    elif type == 'FACE_OVAL':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_FACE_OVAL
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    elif type == 'LIPS':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_LIPS
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    elif type == 'LEFT_EYE':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_LEFT_EYE
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    elif type == 'LEFT_EYEBROW':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_LEFT_EYEBROW
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    elif type == 'LEFT_IRIS':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_LEFT_IRIS
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
    elif type == 'RIGHT_EYE':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_RIGHT_EYE
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    elif type == 'RIGHT_EYEBROW':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_RIGHT_EYEBROW
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    elif type == 'RIGHT_IRIS':
        landmark_drawing_spec = None
        connections = mp_face_mesh.FACEMESH_RIGHT_IRIS
        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
    else:
        print(f'info:標記類型:{type}不正確')
        return
    
    # draw 
    for face_landmarks in result_wrap.mp_result.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=connections,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec)

