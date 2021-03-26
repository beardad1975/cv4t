import cv2
import numpy as np


def pngOverlay(img, pngImg):
    alpha = pngImg[:,:,3] / 255.0
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    
    png_bgr = pngImg[:,:,:3]
    result_img = (png_bgr*alpha_3 + img*(1-alpha_3)) 
    
    # NB: change type to uint8
    print(np.sum(result_img >200))

    return result_img.astype(np.uint8)


def alpha_blit_bgr(img, ):
    pass



# bg = np.full((100,100,3), 255, dtype=np.uint8)
# print(img)
# 
# fg = np.full((100,100,4), 10, dtype=np.uint8)
# fg[:,:,3] = 200
# png_img[:,1,:] = 200
#print(png_img)

bg = cv2.imread("bg.png")
fg = cv2.imread("fg.png", cv2.IMREAD_UNCHANGED)



m = pngOverlay(bg, fg)
cv2.imshow('png', m)



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





