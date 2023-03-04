from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np

import os 
from pathlib import Path




def draw_text(img, text, pos,  font_size , color):
    if img is None or not text :
        print('info: 無影像陣列或文字')
        return
    
    if type(text) is not str:
        text = str(text)

    font = ImageFont.truetype('msjh.ttc', font_size)
    img_pillow = Image.fromarray(img)

    draw = ImageDraw.Draw(img_pillow)
    draw.text(pos,  text, font = font, fill = color)
    
    img.flags.writeable = True
    img[:] = np.array(img_pillow)[:]



# def draw_text(img, text, pos,  font_size , color):
#     if img is None or not text :
#         print('info: 無影像陣列或文字')
#         return
    
#     if type(text) is not str:
#         text = str(text)

#     x = pos[0]
#     y = pos[1]
    
#     img_height, img_width = img.shape[0], img.shape[1]
    
# #     if img.ndim == 3:
# #         img_height, img_width, _ = img.shape
# #     else : # grayscale
# #         img_height, img_width = img.shape
    
#     #check range
#     if not  0 <= x < img_width or not  0 <= y < img_height  :
#         print('info: 文字位置超出範圍')
#         return

#     # get font bitmap
#     font = ImageFont.truetype("msjh.ttc", font_size, encoding="utf-8") 
#     font_bitmap = font.getmask(text)
#     font_width, font_height = font_bitmap.size
#     #print("font: ", font_width, font_height)
#     font_img = np.asarray(font_bitmap, np.uint8)
#     font_img = font_img.reshape( (font_height, font_width))

#     # determine width and height
#     x_right_bound = x + font_width
#     mask_width = font_width
#     if x_right_bound > img_width :
#         x_right_bound = img_width
#         mask_width = img_width - x

#     y_bottom_bound = y + font_height
#     mask_height = font_height
#     if y_bottom_bound > img_height :
#         y_bottom_bound = img_height
#         mask_height = img_height - y
    
#     #print("mask: ", mask_width, mask_height)
    
#     ret , font_mask = cv2.threshold(font_img[:mask_height, :mask_width], 127, 255, cv2.THRESH_BINARY)
    
#     font_mask_inv = 255 - font_mask
    
    
#     if img.ndim == 3:
#         color_img = np.empty((mask_height, mask_width, 3), np.uint8)
#         color_img[:,:] = color
        
#         ori_area = img[y:y_bottom_bound, x:x_right_bound]
        
#         ori_area_masked = cv2.bitwise_and(ori_area, ori_area, mask=font_mask_inv)
#         font_area_masked = cv2.bitwise_and(color_img, color_img, mask=font_mask)
        
#         img[y:y_bottom_bound, x:x_right_bound] = ori_area_masked + font_area_masked
    
#     else: # grayscale
#         color_img = np.empty((mask_height, mask_width), np.uint8)
#         color_img[:] = 255
        
#         ori_area = img[y:y_bottom_bound, x:x_right_bound]
        
#         ori_area_masked = cv2.bitwise_and(ori_area, ori_area, mask=font_mask_inv)
#         font_area_masked = cv2.bitwise_and(color_img, color_img, mask=font_mask)
        
#         img[y:y_bottom_bound, x:x_right_bound] = ori_area_masked + font_area_masked
    
#     return img

def blit_alpha_img(img, alpha_img, pos, anchor_centered=False):
    if img is None or alpha_img is None :
        print('info: 無影像陣列')
        return
    
    #check alpha
    if img.ndim != 3 or img.shape[2] != 3:
        print('info: 陣列不是彩色影像')
        return
    #check alpha
    if alpha_img.ndim != 3 or alpha_img.shape[2] != 4:
        print('info: 含透明度陣列沒有alpha通道')
        return

    
    

    img_height, img_width = img.shape[0], img.shape[1]
    alpha_img_height, alpha_img_width = alpha_img.shape[0], alpha_img.shape[1]

    if anchor_centered :
        # anchor at center position
        x = int(pos[0]) - alpha_img_width//2
        y = int(pos[1]) - alpha_img_height//2
    else:
        # anchor at top-left position
        x = int(pos[0])
        y = int(pos[1])

    #check range
    if not  0 <= x < img_width or not  0 <= y < img_height  :
        #print('info: 貼上位置 超出影像範圍')
        return

    
    
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

###############
def makeRotateMatrix(angle):
    cost = np.cos(np.deg2rad(angle))
    sint = np.sin(np.deg2rad(angle))
    mat = np.array([[cost, sint, 0],
                   [-sint, cost, 0],
                   [0, 0, 1]], dtype=np.float32)
    return mat

def makeTranslateMatrix(dx, dy):
    mat = np.array([[1, 0, dx],
                   [0, 1, dy],
                   [0, 0, 1]], dtype=np.float32)
    return mat

def makeScaleMatrix(scaleX, scaleY):
    mat = np.array([[scaleX, 0, 0],
                   [0, scaleY, 0],
                   [0, 0, 1]], dtype=np.float32)
    return mat

def calculateRotationAngle(a,b):
    """ return rotation angle from vector a to vector b, in degrees.
    Args:
        a : np.array vector. format (x,y)
        b : np.array vector. format (x,y)
    Returns:
        angle [float]: degrees. 0~360
    """
    # y axis reverse
#     a[1] = -a[1]
#     b[1] = -b[1]
    
    unit_vector_1 = a / np.linalg.norm(a)
    unit_vector_2 = b / np.linalg.norm(b)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    angle = angle/ np.pi * 180
    c = np.cross(b,a)
    #print('cross: ', c,  end=" ")
    if c<0:
#         angle +=180
        angle = - angle
    
    return angle


def two_points_transform(ori_img, ori_left_pt, ori_right_pt,
                         dst_img, dst_pt1, dst_pt2):
    height, width, channels = dst_img.shape
    
    # make sure ori pt1 is the left one
#     if ori_left_pt[0] > ori_right_pt[0]:
#         print('exchange ori pt1 pt2')
#         ori_left_pt, ori_right_pt = ori_right_pt, ori_left_pt

    # make sure  pt1 pt2 cant be  same x 
#     if ori_left_pt[0] == ori_right_pt[0]:
#         raise Exception("x of left pt cant be x of right pt")

    # calculate vector
    ori_x_vector = ori_right_pt[0] - ori_left_pt[0]
    ori_y_vector = ori_right_pt[1] - ori_left_pt[1]
    dst_x_vector = dst_pt2[0] - dst_pt1[0]
    dst_y_vector = dst_pt2[1] - dst_pt1[1]

    if ori_left_pt == ori_right_pt or dst_pt1 == dst_pt2:
        raise Exception(" 來源或是目標的點1、2不能相同")


    # determine source alpha
    if channels == 4 :
        has_alpha = True
        b, g, r, alpha = cv2.split(ori_img)
        ori_img = cv2.merge((b, g, r))

    else :
        has_alpha = False



    # calculate scale ratio
    ori_length = np.sqrt(abs(ori_x_vector)**2 +
                           abs(ori_y_vector)**2)
    dst_length = np.sqrt(abs(dst_x_vector)**2 +
                           abs(dst_y_vector)**2)
    scale_ratio = dst_length / ori_length
    #print('scale ratio: ', scale_ratio)

    # calculate angle
    rotation_angle = calculateRotationAngle(
                           np.array([ori_x_vector, ori_y_vector],dtype=np.float32),
                           np.array([dst_x_vector, dst_y_vector],dtype=np.float32))
    #print('rotation angle: ', rotation_angle)
    


    to_ori_M = makeTranslateMatrix(-ori_left_pt[0], -ori_left_pt[1])
    rotate_M = makeRotateMatrix(rotation_angle)
    
    scale_M = makeScaleMatrix(scale_ratio, scale_ratio)
    to_dst_M = makeTranslateMatrix(dst_pt1[0], dst_pt1[1])

    M = np.matmul(rotate_M, to_ori_M)
    M = np.matmul(scale_M, M)
    M = np.matmul(to_dst_M, M)

    affine_img = cv2.warpAffine(ori_img, M[:2, :], (width, height))

    if has_alpha:
        affine_alpha = cv2.warpAffine(alpha, M[:2, :], (width, height))
        b, g, r = cv2.split(affine_img)
        affine_img = cv2.merge((b, g, r, affine_alpha))


    return affine_img
