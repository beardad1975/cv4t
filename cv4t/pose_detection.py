import mediapipe as mp
import cv2 

import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

## Face Detection ==============================

class PoseDetectorWrap():
    def __init__(self, mp_detector):
        self.mp_detector = mp_detector

    def process(self, img):
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_result = self.mp_detector.process(img_rgb)
        img.flags.writeable = True
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_height, img_width, _ = img.shape
        return PoseDetectionResult(mp_result, img_width, img_height)
    
class PoseDetectionResult():
    def __init__(self, mp_result, img_width, img_height):
        self.mp_result = mp_result
        self.img_width = img_width
        self.img_height = img_height

    def __bool__(self):
        return bool(self.mp_result.pose_landmarks)

    def __len__(self):
        if self.mp_result.pose_landmarks:
            return 1
        else:
            print('沒有偵測到姿勢,無資料')
            return 0

class PoseLandmarksInfo():
    def __init__(self, pose_landmarks, result_wrap):
        self.mp_pose_landmarks = pose_landmarks
        self.img_width = result_wrap.img_width
        self.img_height = result_wrap.img_height

    def __getitem__(self, item):
        item = self.mp_pose_landmarks.landmark[item]

        return (math.floor(item.x * self.img_width),
                 math.floor(item.y * self.img_height))

    def visibility(self, 索引):
        return self.mp_pose_landmarks.landmark[索引].visibility
    
    def x(self, 索引):
        tmp_x = self.mp_pose_landmarks.landmark[索引].x
        return math.floor(tmp_x * self.img_width)
    
    def y(self, 索引):
        tmp_y = self.mp_pose_landmarks.landmark[索引].y
        return math.floor(tmp_y * self.img_height)
    
    def z(self, 索引):
        tmp_z = self.mp_pose_landmarks.landmark[索引].z
        return math.floor(tmp_z * self.img_width)

    def 座標3D(self, 索引):
        item = self.mp_pose_landmarks.landmark[索引]
        return (math.floor(item.x * self.img_width),
                 math.floor(item.y * self.img_height),
                 math.floor(item.z * self.img_width))
    
    def 在範圍內嗎(self, 索引):
        item = self.mp_pose_landmarks.landmark[索引]
        tmp_x = math.floor(item.x * self.img_width)
        tmp_y = math.floor(item.y * self.img_height)
        return 0 <= tmp_x < self.img_width and 0 <= tmp_y < self.img_height

        
    def __len__(self):
        if self.mp_pose_landmarks.landmark:
            return len(self.mp_pose_landmarks.landmark)
        else:
            return 0


# class Pose3DLandmarksInfo():
#     def __init__(self, pose_landmarks, result_wrap):
#         self.mp_pose_landmarks = pose_landmarks
#         self.img_width = result_wrap.img_width
#         self.img_height = result_wrap.img_height

#     def __getitem__(self, idx):
#         item = self.mp_pose_landmarks.landmark[idx]
#         return (math.floor(item.x * self.img_width),
#                  math.floor(item.y * self.img_height),
#                  math.floor(item.z * self.img_width))

#     def __len__(self):
#         if self.mp_pose_landmarks.landmark:
#             return len(self.mp_pose_landmarks.landmark)
#         else:
#             return 0


def 設置PoseDetection(最小偵測信心=0.5):
    mp_detector =  mp_pose.Pose(min_detection_confidence=最小偵測信心)
    return PoseDetectorWrap(mp_detector)


def 標記Pose(img, result_wrap):
    # 標記 矩形關鍵點
    if not result_wrap:
        print('info: 沒有偵測到姿勢,不標示')
        return

    mp_drawing.draw_landmarks(
        img,
        result_wrap.mp_result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


def 取出PoseLandmarks(result_wrap):
    if not result_wrap:
        print('info: 沒有偵測到姿勢,無資料')
        return
    
    pose_landmarks = result_wrap.mp_result.pose_landmarks
    return PoseLandmarksInfo(pose_landmarks, result_wrap)

# def 取出Pose3DLandmarks(result_wrap):
#     if not result_wrap:
#         print('info: 沒有偵測到姿勢,無資料')
#         return
    
#     pose_landmarks = result_wrap.mp_result.pose_landmarks
#     return Pose3DLandmarksInfo(pose_landmarks, result_wrap)