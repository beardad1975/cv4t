import cv2
import numpy as np
from imutils import resize

def oldPngOverlay(img, pngImg):
    alpha = pngImg[:,:,3] / 255.0
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    
    png_bgr = pngImg[:,:,:3]
    result_img = (png_bgr*alpha_3 + img*(1-alpha_3)) 
    
    # NB: change type to uint8
    print(np.sum(result_img >200))

    return result_img.astype(np.uint8)


def blit_alpha_img(img, alpha_img, pos):
    if img is None or alpha_img is None :
        print('<< 無影像陣列 >>')
        return
    
    #check alpha
    if img.ndim != 3 or img.shape[2] != 3:
        print('<< 陣列不是彩色影像 >>')
        return
    #check alpha
    if alpha_img.ndim != 3 or alpha_img.shape[2] != 4:
        print('<< 含透明度陣列沒有alpha通道 >>')
        return

    
    x = int(pos[0])
    y = int(pos[1])

    img_height, img_width = img.shape[0], img.shape[1]

    #check range
    if not  0 <= x < img_width or not  0 <= y < img_height  :
        print('<< 位置超出範圍 >>')
        return

    alpha_img_height, alpha_img_width = alpha_img.shape[0], alpha_img.shape[1]
    
    # determine width and height
    x_right_bound = x + alpha_img_width
    blit_width = alpha_img_width
    if x_right_bound > img_width :
        x_right_bound = img_width
        blit_width = img_width - x

    y_bottom_bound = y + alpha_img_height
    blit_height = alpha_img_height
    if y_bottom_bound > img_height :
        y_bottom_bound = img_height
        blit_height = img_height - y
    
    #print("blit: ", x, y ,blit_width, blit_height)    
    
    alpha = alpha_img[:blit_height, :blit_width, 3] / 255.0
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    
    
    alpha_blit_bgr = alpha_img[:blit_height,:blit_width,:3]
    
    img_blit = img[y:y_bottom_bound, x:x_right_bound]
    #print(img_blit.shape)
    result_img = (alpha_blit_bgr*alpha_3 + img_blit *(1-alpha_3))
    # NB: change type to uint8    
    img[y:y_bottom_bound, x:x_right_bound] = result_img.astype(np.uint8)
    
    #print(img_blit)
    
    return img

from pathlib import Path

path = Path(cv2.__file__).parent / 'data' / 'haarcascade_frontalface_alt2.xml'
#print(face_cascade_path)
#face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
#path = 'D:/py4t/env4t/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
#path = 'D:\\py4t\\env4t\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(str(path))


fg = cv2.imread("fg.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)



last_x = 0
last_y = 0

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.08,
                                          minNeighbors=6,
                                          minSize=(100, 100),
                                          maxSize=(300,300),
        )
     
    for x, y ,w, h in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        x = (x + last_x) * 0.5
        y = (y + last_y) * 0.5
        #resized_fg  = resize(fg, width= w)
        
        blit_alpha_img(img , fg, (x,y-h))
        last_x = x
        last_y = y
    
    cv2.imshow('png', img)
    

    cv2.waitKey(1)

#bg = cv2.imread("green_hill.png", cv2.IMREAD_UNCHANGED)
#fg = cv2.imread("1.png", cv2.IMREAD_UNCHANGED)

#print('bg shape:', bg.shape)
#print('fg shape:', fg.shape)

# normalize alpha channels from 0-255 to 0-1
#alpha_bg = bg[:,:,3] / 255.0
#alpha_fg = fg[:,:,3] / 255.0

# # set adjusted colors
# for color in range(0, 3):
#     background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
#         alpha_background * background[:,:,color] * (1 - alpha_foreground)
# 
# # set adjusted alpha and denormalize back to 0-255
# background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
# 
# # display the image
# cv2.imshow("Composited image", background)
# cv2.waitKey(0)





