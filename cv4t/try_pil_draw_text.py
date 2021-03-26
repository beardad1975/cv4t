from PIL import ImageFont
import cv2

import numpy as np


#img = cv2.imread('pic.jpg', cv2.IMREAD_COLOR)

def draw_text(img, text, pos,  font_size , color):
    if img is None or not text :
        print('<< 無影像陣列或文字 >>')
        return
    
    x = pos[0]
    y = pos[1]
    
    img_height, img_width, _ = img.shape
    
    #check range
    if not  0 <= x < img_width or not  0 <= y < img_height  :
        print('<< 文字位置超出範圍 >>')
        return

    # get font bitmap
    font = ImageFont.truetype("msjh.ttc", font_size, encoding="utf-8") 
    font_bitmap = font.getmask(text)
    font_width, font_height = font_bitmap.size
    font_img = np.asarray(font_bitmap, np.uint8)
    font_img = font_img.reshape( (font_height, font_width))

    # determine width and height
    x_right_bound = x + font_width
    mask_width = font_width
    if x_right_bound > img_width :
        x_right_bound = img_width
        mask_width = img_width - x

    y_bottom_bound = y + font_height
    mask_height = font_height
    if y_bottom_bound > img_height :
        y_bottom_bound = img_height
        mask_height = img_height - y
    
    ret , font_mask = cv2.threshold(font_img[:mask_height, :mask_width], 127, 255, cv2.THRESH_BINARY)
    
    font_mask_inv = 255 - font_mask
    
    color_img = np.empty((mask_height, mask_width, 3), np.uint8)
    color_img[:,:] = color
    
    
    
    ori_area = img[y:y_bottom_bound, x:x_right_bound]
    
    ori_area_masked = cv2.bitwise_and(ori_area, ori_area, mask=font_mask_inv)
    font_area_masked = cv2.bitwise_and(color_img, color_img, mask=font_mask)
    
    img[y:y_bottom_bound, x:x_right_bound] = ori_area_masked + font_area_masked
    
    #print(font_mask_inv, font_mask_inv.shape)
    #print(font_area_masked, font_area_masked.shape)
    
    
    #print(color_img, )
    
    #x = pos[0]
    #y = pos[1]
    
    #font_bgr_img = cv2.merge([font_img, font_img, font_img])
    # =  
    
    #cv2.imshow('1',font_img)
    #cv2.waitKey(0)
    
    
    #print(img.shape, img.dtype)
    

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
 
 
    draw_text(img, '你好', (640,16), 50, (23,200,56) )

    cv2.imshow('1',img)
    cv2.waitKey(10)


