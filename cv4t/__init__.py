import cv2
from mss import mss
import imutils
import numpy as np
from . import color
from .draw_lib import draw_text, blit_alpha_img, two_points_transform
#from .dnn import 深度學習人臉模型
from .face_detection import 設置FaceDetection, 標記Face, 取出Face, 取出Face清單, \
    設置FaceMesh, 取出Landmarks, 取出3DLandmarks, 標記FaceMesh  
from .mask_align import load_csv_annotation, mask_transform


__all__ = [ 
            '讀取影像灰階', '讀取影像彩色', '顯示影像', '等待按鍵',
            '關閉全部視窗', '關閉視窗', '儲存影像', '設置影像擷取', '擷取影像',
            '彩色轉灰階', '灰階轉彩色', '左右翻轉', '上下翻轉', '上下左右翻轉',
            '擷取螢幕灰階', '擷取螢幕', '畫矩形', '畫矩形實心', 'color',
            '畫圓形', '畫圓形實心', '旋轉影像', '平移影像', '縮放影像',
            '調整亮度', '調整對比', '模糊', '高斯模糊', '灰階轉黑白',
            'Canny邊緣偵測', '畫出文字', '讀取png影像',
            '畫直線', '畫折線', '設置FaceDetection', '標記Face',
            '取出Face', '取出Face清單', '設置FaceMesh', '取出Landmarks',
            '標記FaceMesh', '取出3DLandmarks', '兩點transform', '貼上png','貼上png中心點',
            '讀取面具對應', '面具transform', 
            ]





### Custom Exceptions
class ImageReadError(Exception):
    def __init__(self, value):
        message = f"<< 無法讀取影像檔 (檔名:{value}) >>"
        super().__init__(message)

class ImageWriteError(Exception):
    def __init__(self, value):
        message = f"<< 無法儲存影像檔 (檔名:{value}) >>"
        super().__init__(message)

class CameraOpenError(Exception):
    def __init__(self, value=''):
        message = f"<< 攝影機開啟錯誤 {value} >>"
        super().__init__(message)     

class CameraReadError(Exception):
    def __init__(self, value=''):
        message = f"<< 攝影機讀取錯誤 {value} >>"
        super().__init__(message)    


### wrapper functions

def 讀取影像灰階(filename):
    ret = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if ret is None:
        raise ImageReadError(filename)
    else:
        return ret

def 讀取影像彩色(filename):
    ret = cv2.imread(filename, cv2.IMREAD_COLOR)
    if ret is None:
        raise ImageReadError(filename)
    else:
        return ret

def 讀取png影像(filename):
    ret = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if ret is None:
        raise ImageReadError(filename)
    else:
        return ret    


def 儲存影像(filename, image):
    ret = cv2.imwrite(filename, image)
    if ret is False:
        ImageWriteError(filename)

def 彩色轉灰階(image):
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def 灰階轉彩色(image):
    if image.ndim == 3:
        return image
    elif image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def 灰階轉黑白(image, 門檻值=128):
    if image.ndim == 3:
        return image
    elif image.ndim == 2:
        ret, result_image = cv2.threshold(image, 門檻值, 255, cv2.THRESH_BINARY)
        return result_image if ret else  image 



def 調整亮度(image, 亮度=0):
    return imutils.adjust_brightness_contrast(
        image, contrast=0, brightness=亮度
        )

def 調整對比(image, 對比=0):
    return imutils.adjust_brightness_contrast(
        image, contrast=對比, brightness=0
        )

def 左右翻轉(image):
    return cv2.flip(image, 1)

def 上下翻轉(image):
    return cv2.flip(image, 0)

def 上下左右翻轉(image):
    return cv2.flip(image, -1)


def 旋轉影像(image, 角度=90):
    return imutils.rotate(image, angle=角度) 

def 平移影像(image, 水平, 垂直):
    return imutils.translate(image, 水平, 垂直)

def 縮放影像(image, 寬度=None, 高度=None):
    if 寬度 is not None and 寬度 >=0:
        return imutils.resize(image, width=寬度)
    elif 高度 is not None and 高度 >=0:
        return imutils.resize(image, height=高度)




def 模糊(image, 核心=5):
    return cv2.blur(image, (核心,核心))

def 高斯模糊(image, 核心=5):
    return cv2.GaussianBlur(image, (核心,核心), 0, 0)

def Canny邊緣偵測(image):
    if image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imutils.auto_canny(image)


def 設置影像擷取(id=0, 解析度=None, 後端=None):

    if 後端 == 'DSHOW':
        cap = cv2.VideoCapture(id, cv2.CAP_DSHOW)
    else:
        # backend auto
        cap = cv2.VideoCapture(id)

    if 解析度 == '720p':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    elif 解析度 == '1080p':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        CameraOpenError()
    return cap

def 擷取影像(cap):
    ret, image = cap.read()
    if ret is False:
        CameraReadError()
    return image

# for screenshot
sct = mss()

def 擷取螢幕(row1, row2, col1, col2):
    global sct

    monitor = {}
    monitor['top']= row1
    monitor['left']= col1
    monitor['width']= col2 - col1
    monitor['height']= row2 - row1
    
    img = np.array(sct.grab(monitor))
    
    return img


def 擷取螢幕灰階(row1, row2, col1, col2):
    global sct

    monitor = {}
    monitor['top']= row1
    monitor['left']= col1
    monitor['width']= col2 - col1
    monitor['height']= row2 - row1
    
    img = np.array(sct.grab(monitor))
    
    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)



active_windows_name_set = set()

def 顯示影像(image, 視窗名稱=None):
    
    if 視窗名稱 is not None:
        if type(視窗名稱) is not str:
            視窗名稱 = str(視窗名稱)
        cv2.imshow(視窗名稱,image)
        active_windows_name_set.add(視窗名稱)
        cv2.waitKey(1)
    else:        
        cv2.imshow('Image 1',image)
        active_windows_name_set.add('Image 1')
        cv2.waitKey(1)

def 等待按鍵(延遲=0):
    ret = cv2.waitKey(延遲)
    if ret == -1:
        return None
    else:
        return chr(ret)

def 關閉全部視窗():
    cv2.destroyAllWindows()

def 關閉視窗(視窗名稱):
    if 視窗名稱 in active_windows_name_set :
        cv2.destroyWindow(視窗名稱)
        active_windows_name_set.remove(視窗名稱)


def 畫直線(image, pt1, pt2, color=(0,0,255), thickness=2):
    if thickness <= 0 : thickness = 2
    if type(thickness) is not int : thickness = int(thickness)
    
    if image.ndim == 2 : color=255
    
    return cv2.line(image, pt1, pt2, color, thickness)


def 畫矩形(image, pt1, pt2, color=(0,0,255), thickness=1):
    if thickness <= 0 : thickness = 1
    if type(thickness) is not int : thickness = int(thickness)
    
    if image.ndim == 2 : color=255
    return cv2.rectangle(image, pt1, pt2, color, thickness)



def 畫矩形實心(陣列, 點1, 點2, 顏色=(0,0,255), 線寬=-1):
    if 陣列.ndim == 2 : 顏色=255
    return cv2.rectangle(陣列, 點1, 點2, 顏色, 線寬)


def 畫圓形(image, center, radius=2, color=(0,0,255), thickness=2 ):
    if image.ndim == 2 : color=255
    if type(radius) is not int : radius = int(radius)
    if type(thickness) is not int : thickness = int(thickness)
    return cv2.circle(image, center,radius, color, thickness )

def 畫圓形實心(陣列, 圓心, 半徑=1, 顏色=(0,0,255), 線寬=-1 ):
    if 陣列.ndim == 2 : 顏色=255
    if type(半徑) is not int : 半徑 = int(半徑)
    if type(線寬) is not int : 線寬 = int(線寬)
    return cv2.circle(陣列, 圓心, 半徑, 顏色, 線寬 )

def 畫折線(陣列, 點清單, 顏色=(0,0,255), 線寬=2, 封閉=False):
    if type(線寬) is not int : 線寬 = int(線寬)
    points = np.array(點清單, np.int32)
    return cv2.polylines(陣列, [points], 封閉, 顏色, 線寬)



def 畫出文字(image, text, pos, size=30, color=(0,0,255)):
    return draw_text(image, text, pos , size, color)

# def 畫出文字(image, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255),thickness=1):
#      return cv2.putText(image, text, pos, font, fontScale, color, thickness)


# def 貼上去背影像(影像, 去背影像, 位置):
#     return blit_alpha_img(影像, 去背影像, 位置)

def 兩點transform(來源影像, 來源pt1, 來源pt2, 目標影像, 目標pt1, 目標pt2):
    return two_points_transform(來源影像, 來源pt1, 來源pt2, 目標影像, 目標pt1, 目標pt2)

def 貼上png中心點(img, alpha_img, pos=(0,0)):
    return blit_alpha_img(img, alpha_img, pos, anchor_centered=True)

def 貼上png(img, alpha_img, pos=(0,0)):
    return blit_alpha_img(img, alpha_img, pos)

def 讀取面具對應(csv檔名, 面具影像):
    return load_csv_annotation(csv檔名, 面具影像)

def 面具transform(來源影像, 來源對應, 目標影像, 偵測結果):
    return mask_transform(來源影像, 來源對應, 目標影像, 偵測結果)



if __name__ == '__main__' :
    pass
    
