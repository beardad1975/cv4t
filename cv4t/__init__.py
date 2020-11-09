import cv2



__all__ = [ 
            '讀取灰階影像', '讀取彩色影像', '顯示影像', '等待按鍵',
            '關閉所有影像',


            ]

class ImageReadError(Exception):
    def __init__(self, value):
        message = f"無法讀取影像檔 (檔名:{value})"
        super().__init__(message)




def 讀取灰階影像(filename):
    ret = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if ret is None:
        raise ImageReadError(filename)
    else:
        return ret

def 讀取彩色影像(filename):
    ret = cv2.imread(filename, cv2.IMREAD_COLOR)
    if ret is None:
        raise ImageReadError(filename)
    else:
        return ret

win_name_prefix = '影像'
win_name_counter = 0

def 顯示影像(image, 新視窗=False):
    global win_name_prefix, win_name_counter    
    if 新視窗:
        win_name_counter += 1
    win_name = win_name_prefix + str(win_name_counter)
    cv2.imshow(win_name,image)
    cv2.waitKey(1)

def 等待按鍵(延遲=0):
    ret = cv2.waitKey(延遲)
    if ret == -1:
        return None
    else:
        return chr(ret)

def 關閉所有影像():
    cv2.destroyAllWindows()




if __name__ == '__main__' :
    pass
    
