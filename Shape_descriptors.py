import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.interpolate as it


def fft_shape(img,n_fft):
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, imbw = cv.threshold(img,30,255,cv.THRESH_BINARY)
    
    # Run findContours - Note the RETR_EXTERNAL flag
    # Also, we want to find the best contour possible with CHAIN_APPROX_NONE
    contours, hierarchy = cv.findContours(imbw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if len(contours) > 1:
        s = 0
        for i in range(len(contours)):
            if contours[i].shape[0] > s:
                s = contours[i].shape[0]
                max_ele = i
        contours = [contours[max_ele]]            
    
    # Create an output of all zeroes that has the same shape as the input
    # image
    out = np.zeros_like(img)
    
    # On this output, draw all of the contours that we have detected
    # in white, and set the thickness to be 3 pixels
    cv.drawContours(out, contours, -1, 255, 3)
    
    # # Spawn new windows that shows us the donut
    # # (in grayscale) and the detected contour
    # cv.imshow('Donut', img) 
    # cv.imshow('Output Contour', out)
    
    # trouver le centroid
    cnt = contours[0]
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    # fonction d'interpolation
    cntr = (cnt.reshape(cnt.shape[0],cnt.shape[2])).T
    f = it.interp1d(np.arange(0,len(cnt)), cntr)
    
    # Changer le bords en 64 points
    cnt64 = f(np.linspace(0,len(cnt),n_fft+1)[:-1])
    
    # centroid distance
    cenDist = ((cnt64[0,:] - cx)**2 + (cnt64[1,:] - cy)**2)**0.5
    cenDistn = (cenDist - min(cenDist)) / (max(cenDist) - min(cenDist))
    
    # transformation discrete de Fourier
    fft = np.fft.fft(cenDistn)
    
    return np.abs(fft[1:int(fft.shape[0]/2)])