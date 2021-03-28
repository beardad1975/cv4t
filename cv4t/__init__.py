import cv2
from mss import mss
import numpy as np
from . import color

__all__ = [ 
            '讀取影像灰階', '讀取影像彩色', '顯示影像', '等待按鍵',
            '關閉所有影像', '儲存影像', '設置影像擷取', '擷取影像',
            '彩色轉灰階', '灰階轉彩色', '左右翻轉', '上下翻轉', '上下左右翻轉',
            '擷取螢幕灰階', '擷取螢幕', '畫方形', '畫實心方形', 'color',
            '畫圓形', '畫實心圓形',
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

def 左右翻轉(image):
    return cv2.flip(image, 1)

def 上下翻轉(image):
    return cv2.flip(image, 0)

def 上下左右翻轉(image):
    return cv2.flip(image, -1)


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





def 顯示影像(image, 視窗名稱=None):
    global win_name_counter    
    
    if 視窗名稱 is not None:
        cv2.imshow(視窗名稱,image)
        cv2.waitKey(1)
    else:        
        cv2.imshow('1',image)
        cv2.waitKey(1)

def 等待按鍵(延遲=0):
    ret = cv2.waitKey(延遲)
    if ret == -1:
        return None
    else:
        return chr(ret)

def 關閉所有影像():
    cv2.destroyAllWindows()


def 畫方形(image, x, y, 寬, 高, 顏色=(0,0,255), 線寬=2):
    if 線寬 <= 0 : 線寬 = 2
    if image.ndim == 2 : 顏色=255
    return cv2.rectangle(image, (x, y), (x+寬,y+高), 顏色, 線寬)

def 畫實心方形(image, x, y, 寬, 高, 顏色=(0,0,255), 線寬=-1):
    if image.ndim == 2 : 顏色=255
    return cv2.rectangle(image, (x, y), (x+寬,y+高), 顏色, 線寬)


def 畫圓形(image, x, y, 半徑, 顏色=(0,0,255), 線寬=2 ):
    if image.ndim == 2 : 顏色=255
    return cv2.circle(image, (x,y),半徑, 顏色, 線寬 )

def 畫實心圓形(image, x, y, 半徑, 顏色=(0,0,255), 線寬=-1 ):
    if image.ndim == 2 : 顏色=255
    return cv2.circle(image, (x,y),半徑, 顏色, 線寬 )

if __name__ == '__main__' :
    pass
    
