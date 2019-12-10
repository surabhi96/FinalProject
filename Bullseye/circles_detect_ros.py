#!/usr/bin/env python

from __future__ import print_function
import sys
import os
PY3 = sys.version_info[0] == 3
dirpath = os.getcwd()

if PY3:
    xrange = range
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy

import numpy as np
import cv2
import imutils


bridge = CvBridge()
mask_pub = rospy.Publisher('/mask', Image, queue_size=1)
def find_circles(my_img):
    img = bridge.imgmsg_to_cv2(my_img, "bgr8")
    
    img_orig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img_orig = clahe.apply(img_orig)
    img = cv2.medianBlur(img_orig,3)
    ret,thresh_binary = cv2.threshold(img,210,255,cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh_binary, np.ones((3,3), np.uint8), iterations=1)

    im, cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((dilate.shape[0], dilate.shape[1], 3), np.uint8)
 
    # # draw contours 
    for i in range(len(cnts)):
        color_contours = (255, 0, 255) 
    
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

            cv2.drawContours(fresh_im, cnts, cnt_num[len(cnt_num)-1-i], (255, 255, 255), -1)
           
    mask = cv2.bitwise_and(img_orig, img_orig, mask = np.uint8(fresh_im))

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
            # cv2.imshow("Detected Circle", mask) 
            # cv2.waitKey(0) 
    mask_pub.publish(bridge.cv2_to_imgmsg(mask, "mono8")) 

     
def main():
    rospy.init_node('bullseye_detection', anonymous=True)
    rospy.Subscriber('/duo3d/left/image_rect_throttle', Image, find_circles)
    rospy.spin()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
