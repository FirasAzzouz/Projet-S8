# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:41:49 2019

@author: FANGXU
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as it

def outerContours(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, imbw = cv.threshold(img,30,255,cv.THRESH_BINARY)
    
    # dilation and erosion
    kernel = np.ones((5,5),np.uint8)
    imbw = cv.dilate(imbw,kernel,iterations = 1)
    imbw = cv.erode(imbw,kernel,iterations = 1)
    
    # Run findContours - Note the RETR_EXTERNAL flag
    # Also, we want to find the best contour possible with CHAIN_APPROX_NONE
    contours, hierarchy = cv.findContours(imbw.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    # Prendre le contour le plus long si plusieurs sont identifiés
    if len(contours) > 1:
        s = 0
        for i in range(len(contours)):
            if contours[i].shape[0] > s:
                s = contours[i].shape[0]
                max_ele = i
        contours = [contours[max_ele]] 
    return contours

def shapeFeatures(img, n_fft = 64):
           
    contours = outerContours(img)
    
    # calculer les moments
    cnt = contours[0]
    M = cv.moments(cnt)
    
    # features simples
    area = M['m00']
    peri = cv.arcLength(cnt,True)
    cirRatio = 4*np.pi*area/peri**2 # feature 1
    coef = np.polyfit(cnt[:,0,0],cnt[:,0,1],1) # feature 2, best fit line, highest power first
    width, height = cv.minAreaRect(cnt)[1]
    if height > width:
        temp = width
        width = height
        height = temp
    ecc = (1 - (height/width)**2)**0.5 # feature 3
    rect = area / (width*height) # feature 4
    
    # trouver le centroid
    cx = int(M['m10']/M['m00']) 
    cy = int(M['m01']/M['m00'])
    
    # fonction d'interpolation
    cntr = (cnt.reshape(cnt.shape[0],cnt.shape[2])).T
    dist = cntr[:,np.mod(range(1,len(cnt)+1),len(cnt))] - cntr
    dist = np.cumsum(np.linalg.norm(dist, axis=0))
    dist = np.concatenate(([0],dist))
    f = it.interp1d(dist, cntr[:,np.mod(range(0,len(cnt)+1),len(cnt))])
    
    # Changer le bords en 64 points
    cnt64 = f(np.linspace(0,dist[-1],n_fft+1)[:-1])
    
    # centroid distance
    cenDist = ((cnt64[0,:] - cx)**2 + (cnt64[1,:] - cy)**2)**0.5
    cenDistn = (cenDist - min(cenDist)) / (max(cenDist) - min(cenDist))
    
    # cumulative angular function
    w = n_fft // 16
    if w == 0: 
        w += 1
    theta = np.arctan2((cnt64[1,:]-cnt64[1,range(-w,n_fft-w)]),\
                       (cnt64[0,:]-cnt64[0,range(-w,n_fft-w)]))
    
    K = np.mod(theta[:] - theta[range(-1,n_fft - 1)] + np.pi,2*np.pi) - np.pi
    
    # contour area
    conArea = []
    for i in range(n_fft):
        #cnti = (np.array([[cx,cy],cnt64[:,i-1],cnt64[:,i]])).reshape(3,1,2)
        a = cx*cnt64[1,i-1] + cnt64[0,i-1]*cnt64[1,i] + cnt64[0,i]*cy \
            - (cy*cnt64[0,i-1] + cnt64[1,i-1]*cnt64[0,i] + cnt64[1,i]*cx)
        conArea.append(-a)    
    conArea = np.array(conArea) / area
    
    # Average Bending Energy
    BE = np.mean(K**2) # feature 5
    
    # transformation discrete de Fourier
    fft = np.fft.fft(cenDistn) # centroid distance
    fft2 = np.fft.fft(K) # courbature
    fft3 = np.fft.fft(conArea) # contour Area
    
    # rendre un total de 8 caractéristiques dans un seul vector, 
    # dont coef contient 2 elements, les ffts contiennent n_fft/2 elements
    return cirRatio, coef, ecc, rect, BE, \
        np.abs(fft[0:int(n_fft/2)]), \
        np.abs(fft2[0:int(n_fft/2)]), \
        np.abs(fft3[0:int(n_fft/2)])

# rendre les 8 caractéristiques dans un seul vector

def concatShapeFeatures(img, n_fft = 64):
    cirRatio, coef, ecc, rect, BE, fft, fft2, fft3= shapeFeatures(img,n_fft)
    output = np.concatenate(([cirRatio], coef, [ecc], [rect], [BE], fft, fft2, fft3))
    return output

# tester le code
if __name__ == '__main__':
    
    filename = 'datasets/aloi_red4_stereo/534/534_c.png'
    img = cv.imread(filename)
    
    _, _, _, _, _, fft, fft2, fft3 = shapeFeatures(img)
    
    # find contours
    contours = outerContours(img)
    
    # pre-treatment
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Create an output of all zeroes that has the same shape as the input
    # image
    out = np.zeros_like(img)
    
    # On this output, draw all of the contours that we have detected
    # in white, and set the thickness to be 3 pixels
    cv.drawContours(out, contours, -1, 255, 3)
    
    # Spawn new windows that shows us the donut
    # (in grayscale) and the detected contour
    #cv.imshow('Donut', img) 
    #cv.imshow('Output Contour', out)
    
    #plt.plot(np.arange(0,32),np.abs(fft))
    plt.bar(np.arange(0,32),np.abs(fft2))
    #plt.plot(np.arange(0,32),np.abs(fft3))
