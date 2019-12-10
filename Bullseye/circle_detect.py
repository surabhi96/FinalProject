#!/usr/bin/env python

from __future__ import print_function
import sys
import os
PY3 = sys.version_info[0] == 3
dirpath = os.getcwd()

if PY3:
    xrange = range

import numpy as np
import cv2
import imutils

def find_circles(img):
    # gra = np.zeros(img.shape)
    # gray = cv2.normalize(gra, gra, 0, 255, cv2.NORM_MINMAX)

    

    img_orig=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img_orig = clahe.apply(img_orig)
    img = cv2.medianBlur(img_orig,3)
    ret,thresh_binary = cv2.threshold(img,210,255,cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh_binary, np.ones((3,3), np.uint8), iterations=1)
    # erode = cv2.erode(thresh_binary,np.ones((3,3), np.uint8),iterations=1)
    # gray_new = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # # edges are preserved while doing median blur while removing noise 
    # gra = cv2.medianBlur(gray_new,5) 
    # #image normalization
    # 

    # # adaptive threshold 
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,17,2)
    # # erode out the noise
    # thresh = cv2.erode(thresh,np.ones((3,3), np.uint8),iterations=1)

    im, cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((dilate.shape[0], dilate.shape[1], 3), np.uint8)
 
    # # draw contours 
    for i in range(len(cnts)):
        color_contours = (255, 0, 255) 
        # draw contours in a black image
        # cv2.drawContours(drawing, cnts, i, color_contours, 1, 8, hierarchy)
    
    # # do dilation after finding the 
    # drawing1 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=9)
    # drawing10 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=2)
    # drawing11 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=3)
    # drawing12 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=4)
    # drawing13 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=5)
    # drawing14 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=6)
    # drawing15 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=7)
    # drawing16 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=8)
    # drawing17 = cv2.dilate(drawing, np.ones((3,3), np.uint8), iterations=1)

    # img_not = np.zeros((drawing1.shape[0], drawing1.shape[1], 3), np.uint8)
    # img_not = cv2.bitwise_not(drawing1)
    # mask = cv2.cvtColor(img_not, cv2.COLOR_BGR2GRAY)
    # im1, cnts1, hierarchy1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_area = []
    cnt_num = []
    for c in cnts:
        cnt_area.append(cv2.contourArea(c))

    cnt_num = np.argsort(cnt_area)
    cnt_area.sort()
    # print(cnt_area)
    # large_cnts = np.zeros(np.shape(mask))
    fresh_im = np.zeros(np.shape(img_orig))

    


    for i in range(5): # in the 5 largest contours, check if cnt_area > 5000
        if cnt_area[len(cnt_area)-1-i] > 1000:

            # rect = cv2.minAreaRect(cnts[cnt_num[len(cnt_num)-1-i]])
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(fresh_im,[box],0,(0,0,255),2)

            
            
            cv2.drawContours(fresh_im, cnts, cnt_num[len(cnt_num)-1-i], (255, 255, 255), -1)
            # im_temp = 255*np.ones(mask.shape) - fresh_im
            # cv2.drawContours(large_cnts, cnts1, cnt_num[len(cnt_num)-1-i], (255, 255, 255), -1)

    # # dilate large conoturs 
    # large_cnts = cv2.dilate(large_cnts, np.ones((5,5), np.uint8), iterations=1)

    mask = cv2.bitwise_and(img_orig, img_orig, mask = np.uint8(fresh_im))

    # edges = cv2.Canny(mask,50,150,apertureSize = 3)
    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

    
    # im, contours, h = cv2.findContours(mask,1,2)
    # mask_new=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

    # for contour in contours:
    #     # (x,y,w,h) = cv2.boundingRect(contour)
    #     # cv2.rectangle(mask_new, (x,y), (x+w,y+h), (0,255,0), 2)
 
    #     rect = cv2.minAreaRect(contour)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(mask_new,[box],0,(0,0,255),2)


    detected_circles = cv2.HoughCircles(mask,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 1, maxRadius = 40) 
  
    # Draw circles that are detected. 
    if detected_circles is not None: 
  
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
      
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
      
            # Draw the circumference of the circle. 
            cv2.circle(mask, (a, b), r, (0, 255, 0), 2) 
      
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(mask, (a, b), 1, (0, 0, 255), 3) 
            cv2.imshow("Detected Circle", mask) 
            cv2.waitKey(0) 

    
    # # cv2.imshow('Adaptive thresh',thresh)
    # # cv2.imshow('Contours after binary thresh',drawing)
    # # cv2.imshow('regions of interest',new_gray)
   
    # cv2.imshow('thresh_binary',mask)
   
    # cv2.waitKey(0)
     
def main():
    # img = cv2.imread('frame0000.jpg')
  
    # find_squares(img)
    for subdir, dirs, files in os.walk(dirpath + '/images_bullseye'):
        files.sort()
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):

                # print(file)
                img = cv2.imread(filepath)
  
                find_circles(img)
                
    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
