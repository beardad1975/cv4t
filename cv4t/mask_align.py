import mediapipe as mp
import cv2
import math
import numpy as np
import csv

############################ load csv annotation


def load_annotate_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
#             print(points)
        return points

def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex

# Calculate Delaunay triangles for set of points
# Returns the vector of indices of 3 points for each triangle
def calculateDelaunayTriangles(rect, points):

  # Create an instance of Subdiv2D
  subdiv = cv2.Subdiv2D(rect)

  # Insert points into subdiv
  for p in points:
    subdiv.insert((int(p[0]), int(p[1])))

  # Get Delaunay triangulation
  triangleList = subdiv.getTriangleList()

  # Find the indices of triangles in the points array
  delaunayTri = []

  for t in triangleList:
    # The triangle returned by getTriangleList is
    # a list of 6 coordinates of the 3 points in
    # x1, y1, x2, y2, x3, y3 format.
    # Store triangle as a list of three points
    pt = []
    pt.append((t[0], t[1]))
    pt.append((t[2], t[3]))
    pt.append((t[4], t[5]))

    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
      # Variable to store a triangle as indices from list of points
      ind = []
      # Find the index of each vertex in the points list
      for j in range(0, 3):
        for k in range(0, len(points)):
          if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
            ind.append(k)
        # Store triangulation as a list of indices
      if len(ind) == 3:
        delaunayTri.append((ind[0], ind[1], ind[2]))

  return delaunayTri


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def load_csv_annotation(pathname, png_img):
    filter_runtime = {}

    annotate_points = load_annotate_landmarks(pathname)

    filter_runtime['annotate_points'] = annotate_points

    hull, hullIndex = find_convex_hull(annotate_points)

    # Find Delaunay triangulation for convex hull points
    sizeImg1 = png_img.shape
    rect = (0, 0, sizeImg1[1], sizeImg1[0])
    dt = calculateDelaunayTriangles(rect, hull)

    filter_runtime['hull'] = hull
    filter_runtime['hullIndex'] = hullIndex
    filter_runtime['dt'] = dt


    return filter_runtime

############################ mask transform

# detect facial landmarks in image via (face_mesh) 
def getMpLandmarks(dst_img, detection):
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = dst_img.shape[:-1]

    # if not results.multi_face_landmarks:
    #     #print('Face not detected!!!')
    #     return 0

    for face_landmarks in detection.mp_result.multi_face_landmarks:
        values = face_landmarks.landmark
        
        relevant_keypnts = []
        for i in selected_keypoint_indices:
            keypoint = (round(values[i].x * width),
                        round(values[i].y * height))
            relevant_keypnts.append(keypoint)
        return relevant_keypnts


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    
    
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    

    img2Rect = img2Rect * mask

    try:
        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
    except ValueError:
        # skip error when face near border
        pass

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):

  # Given a pair of triangles, find the affine transform.
  warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

  # Apply the Affine Transform just found to the src image
  dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

  return dst


def mask_transform(ori_img, filter_runtime, dst_img, detection):
    mp_points_75 = getMpLandmarks(dst_img, detection)

    ori_height, ori_width, channels = ori_img.shape

    if channels != 4 :
        raise ValueError('來源影像需為png去背圖')

    b, g, r, a = cv2.split(ori_img)
    img = cv2.merge((b,g,r))
    #annotate_points = filter_runtime['annotate_points']
    img_alpha = a

    hullIndex = filter_runtime['hullIndex']
    dt = filter_runtime['dt']
    hull1 = filter_runtime['hull']

    # create copy of frame
    #warped_img = np.copy(frame)
    warped_img = np.zeros(dst_img.shape, dtype=np.uint8)

    # Find convex hull
    hull2 = []
    for i in range(0, len(hullIndex)):
        hull2.append(mp_points_75[hullIndex[i][0]])

    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1],3), dtype=np.float32)
    #mask1 = cv2.merge((mask1, mask1, mask1))
    img_alpha_mask = cv2.merge((img_alpha, img_alpha, img_alpha))
    #img_alpha_mask = img_alpha

    # Warp the triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img, warped_img, t1, t2)
        warpTriangle(img_alpha_mask, mask1, t1, t2)

    # Blur the mask before blending
#     mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
    b, g, r = cv2.split(warped_img)
    a, _, _ = cv2.split(mask1)
    a = np.uint8(a)
    #print(np.max(img_alpha_mask))
#         print(warped_img.shape, mask1.shape)
#         print(warped_img.dtype, mask1.dtype)
    output = cv2.merge((b, g, r, a))
    return output