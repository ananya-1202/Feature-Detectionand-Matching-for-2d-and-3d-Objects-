import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

image =[ r"1.jpg"  , 
    r"2.jpg"  ,  
    r"3.jpg" , 
    r"4.jpg" , 
    r"5.jpg"
]
#taking the images at 72 degree to train the image at 5 points then excuting the code in the similar way to 2d feature detection and matching to the image which has
#the most similar features 

sift = cv2.xfeatures2d.SIFT_create()

def interestPoints(add):
    img = cv2.imread(add)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kp,des=sift.detectAndCompute(img,None)
    imgkp=cv2.drawKeypoints(img,kp,img)
    return kp, des  , imgkp

k = []
d = []
im = []

for i in range(len(image)):
    k_ , d_ , i_ = interestPoints(image[i])
    k.append(k_)
    d.append(d_)
    im.append(i_)
    #print(i)


kimg, dimg, iimg = interestPoints(image[1])

m = cv2.imread(image[1])

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
search_params = dict(checks=50)


for i in range(len(image)):
    flann = cv2.FlannBasedMatcher(index_params,search_params) #
    matches = flann.knnMatch(d[i], dimg, k=2)  
    print(len(matches),len(d[i]),len(dimg))

    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6*n.distance:
            matchesMask[i] = [1, 0]
            good.append(m) 


#calculation of the score

'''



img1kp=cv2.drawKeypoints(img1,kp1,img1)
img2kp=cv2.drawKeypoints(img2,kp2,img1)
#cv2.imwrite('m_img1.jpg',img1kp)
#cv2.imwrite('m_img2.jpg',img2kp)

#bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params) #
matches = flann.knnMatch(des1, des2, k=2)  

matchesMask = [[0, 0] for i in range(len(matches))]
good = []
for i, (m, n) in enumerate(matches):
    if m.distance < 0.6*n.distance:
        matchesMask[i] = [1, 0]
        good.append(m) 

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
origin = [0,0]
'''

'''
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

h,w,c = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)


print(dst)

img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)    

'''


cv2.imshow('final',m)
cv2.waitKey(0)
cv2.destroyAllWindows()
