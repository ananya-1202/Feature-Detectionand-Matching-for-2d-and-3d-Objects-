import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

from math import atan2,degrees
import math 

def AngleBtw2Points(pointA, pointB):
  changeInX = pointB[0] - pointA[0]
  changeInY = pointB[1] - pointA[1]
  return degrees(atan2(changeInY,changeInX)) #remove degrees if you want your answer in radians


# returns square of distance b/w two points 
def lengthSquare(X, Y): 
	xDiff = X[0] - Y[0] 
	yDiff = X[1] - Y[1] 
	return xDiff * xDiff + yDiff * yDiff 
	
def printAngle(A, B): 

    C = [0,0]

    # Square of lengths be a2, b2, c2 
    a2 = lengthSquare(B, C) 
    b2 = lengthSquare(A, C) 
    c2 = lengthSquare(A, B) 

    # length of sides be a, b, c 
    a = math.sqrt(a2); 
    b = math.sqrt(b2); 
    c = math.sqrt(c2); 

    # From Cosine law 
    alpha = math.acos((b2 + c2 - a2) /
                (2 * b * c)); 
    betta = math.acos((a2 + c2 - b2) /
                (2 * a * c)); 
    gamma = math.acos((a2 + b2 - c2)/
                (2 * b * a))

    # Converting to degree 
    alpha = alpha * 180 / math.pi;  
    betta = betta * 180 / math.pi; 
    gamma = gamma * 180 / math.pi; 

    return alpha , betta , gamma 
		


vid = cv2.VideoCapture(0) 
add = r"" #address of the training image 
img = cv2.imread(add)
def image_detect_and_compute(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img_building = cv2.imread(img_name)
    img_building = cv2.cvtColor(img_building, cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img_building, None)
    img_kp = cv2.drawKeypoints(img_building, kp, img_building)
    return img_building, kp, des
def image_detect_and_compute_video(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img_building = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img_building, None)
    img_kp = cv2.drawKeypoints(img_building, kp, img_building)
    return img_building, kp, des

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    


    # Display the resulting frame 
    sift = cv2.xfeatures2d.SIFT_create()
    img1, kp1, des1 = image_detect_and_compute(sift, add)
    img2, kp2, des2 = image_detect_and_compute_video(sift, frame)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    allPoints = []
    good = []
    point = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.55*n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)
            point.append(i)

    no = []
    for i, (m, n) in enumerate(matches):
        allPoints.append(n) #interest points in the training image 
        no.append(i)

    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in allPoints ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    origin = [0,0]

    if(len(good)>10):
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w,c = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        print(dst)
        #frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    '''
    l = len(point)-2
    l=3
    if(len(dst)>3):
    
    
        dst_co1 = [dst[0][0][0],dst[0][0][1]] 
        dst_co2 = [dst[l][0][0],dst[l][0][1]]
        src_co1 = [src[point[0]][0][0],src[point[0]][0][1]]
        src_co2 = [src[point[l]][0][0],src[point[l]][0][1]]
        origin = [0,0]
        import math
        from math import atan2,degrees
        alpha1 = (math.atan2(origin[1] - src_co1[1], origin[0] - src_co1[0]) - math.atan2(src_co2[1] - src_co1[1], src_co2[0] - src_co1[0]))
        alpha2 = (math.atan2(origin[1] - src_co2[1], origin[0] - src_co2[0]) - math.atan2(src_co1[1] - src_co2[1], src_co1[0] - src_co2[0]))

        alpha1,alpha2, theta = printAngle(src_co1, src_co2)

        print(alpha1 , alpha2 , theta )

        # This code is contributed 
        # by ApurvaRaj 


        import math
        from math import atan2,degrees
        ratio = ((((dst_co1[0] - dst_co2[0])**2) + ((dst_co1[1]-dst_co2[1])**2) )**0.5)/((((src_co1[0] - src_co2[0])**2) + ((src_co1[1]-src_co2[1])**2) )**0.5)
        #ratio = math.dist(dst_co1,dst_co2)/math.dist(src_co1,src_co2)#ratio of the difference in the distance 
        ratioX = abs((dst_co1[0] - dst_co2[0])/(src_co1[0] - src_co2[0]))
        ratioY = abs((dst_co1[1] - dst_co2[1])/(src_co1[1] - src_co2[1]))

        angle1 = AngleBtw2Points(dst_co1,dst_co2)
        angle2 = AngleBtw2Points(src_co1,src_co2) 
        angle3 = AngleBtw2Points(src_co1,[0,0])
            

        h,w,c = img.shape
        add1 = r"D:\Ananya\3RDi_LABs\Feature_Extraction_and_Matching\Feature_Detection\flower.jpg"
        up = cv2.imread(add1)
        up = cv2.resize(up,(int(w*ratioX),int(h*ratioY)))
        rows,cols,channels = up.shape
        #frame[0:rows, 0:cols] = up
        
        print(frame.shape)
        l = h*ratioY
        b = w*ratioX

        print(l,b,w,h, ratioY, ratioX)
        #frame[0:rows, 0:cols] = up
    '''
    #frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

vid.release() 
cv2.destroyAllWindows() 
