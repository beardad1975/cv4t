from PIL import ImageFont
import cv2
import numpy as np

def draw_text(img, text, pos,  font_size , color):
    if img is None or not text :
        print('<< 無影像陣列或文字 >>')
        return
    
    x = pos[0]
    y = pos[1]
    
    img_height, img_width = img.shape[0], img.shape[1]
    
#     if img.ndim == 3:
#         img_height, img_width, _ = img.shape
#     else : # grayscale
#         img_height, img_width = img.shape
    
    #check range
    if not  0 <= x < img_width or not  0 <= y < img_height  :
        print('<< 文字位置超出範圍 >>')
        return

    # get font bitmap
    font = ImageFont.truetype("msjh.ttc", font_size, encoding="utf-8") 
    font_bitmap = font.getmask(text)
    font_width, font_height = font_bitmap.size
    #print("font: ", font_width, font_height)
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
    
    #print("mask: ", mask_width, mask_height)
    
    ret , font_mask = cv2.threshold(font_img[:mask_height, :mask_width], 127, 255, cv2.THRESH_BINARY)
    
    font_mask_inv = 255 - font_mask
    
    
    if img.ndim == 3:
        color_img = np.empty((mask_height, mask_width, 3), np.uint8)
        color_img[:,:] = color
        
        ori_area = img[y:y_bottom_bound, x:x_right_bound]
        
        ori_area_masked = cv2.bitwise_and(ori_area, ori_area, mask=font_mask_inv)
        font_area_masked = cv2.bitwise_and(color_img, color_img, mask=font_mask)
        
        img[y:y_bottom_bound, x:x_right_bound] = ori_area_masked + font_area_masked
    
    else: # grayscale
        color_img = np.empty((mask_height, mask_width), np.uint8)
        color_img[:] = 255
        
        ori_area = img[y:y_bottom_bound, x:x_right_bound]
        
        ori_area_masked = cv2.bitwise_and(ori_area, ori_area, mask=font_mask_inv)
        font_area_masked = cv2.bitwise_and(color_img, color_img, mask=font_mask)
        
        img[y:y_bottom_bound, x:x_right_bound] = ori_area_masked + font_area_masked
    
    return img

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