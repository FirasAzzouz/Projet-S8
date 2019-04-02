# author: Azzouz
# -*- coding: utf-8 -*-
"""
This file contains functions that compute color descriptors.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import cv2 as cv

# ******************************************************Color Moments descriptor*******************************************************
""" 
This function computes the color moments (average, variance, standard deviation, skewness and kurtosis) of an image for the three color channels. 
@param X: 3-D Array: representation of an image in a color space
@return: 1-D Array: the filtered color histogram 
"""
def moments_calcul(X,color_space):

    if color_space=='hsv':
        X=cv.cvtColor(X, cv.COLOR_BGR2HSV)
    X0=X[:,:,0].reshape(-1)
    X1=X[:,:,1].reshape(-1)
    X2=X[:,:,2].reshape(-1)
    Mean_=np.array([np.mean(X0),np.mean(X1),np.mean(X2)])
    Var_=np.array([np.var(X0),np.var(X1),np.var(X2)])
    Str_=np.array([np.std(X0),np.std(X1),np.std(X2)])
    Skewness_=np.array([skew(X0),skew(X1),skew(X2)])
    Kurtosis_=np.array([kurtosis(X0),kurtosis(X1),kurtosis(X2)])
    
    return np.hstack((Mean_,Var_,Str_,Skewness_,Kurtosis_))

# *****************************************************Color Histogram descriptor***************************************************
""" 
This function computes the color histogram of an image X in a color space using the specified bins.
@param X: 3D-Array: representation of an image in a color space
@param color_space: String : color space, typically "rgb" or "hsv"
@param bins: list of 3 integers: the chosen bins to define the colors. 
Each value represents the number of intervals to which each color channel is divided.
@return: 1-D Array: the color histogram 
"""
def color_histogram(X,color_space,bins):


    if color_space=='rgb':
        lim_colors=[256,256,256]
    elif color_space=='hsv':
        lim_colors=[180,256,256]
        X=cv.cvtColor(X, cv.COLOR_BGR2HSV)
        
    else:
        print("select please a color space among rgb and hsv")
    
    hist = cv.calcHist([X], [0, 1, 2],None, bins , [0, lim_colors[0], 0, lim_colors[1], 0, lim_colors[2]])
    # plt.bar(np.arange(int(bins[0]*bins[1]*bins[2])),hist.flatten())
    # plt.show()
    return hist.flatten()

""" 
This function removes the components thr dominant colors which correspond mainly to the black background.
This allows to compare the images with respect to the colors inside the object. The constant 500 is chosen by visualizing the color histograms
@param hist: 1-D Array representing the color histogram 
@return: 1-D Array containing the filtered color histogram 
"""
def histogram_filter(hist):
  
    for i in range(len(hist)):
        if hist[i]>500:
            hist[i]=0
    return hist

# *****************************************************Color Correlogram descriptor***************************************************
"""
This function finds the neighbors of pixel p at a certain distance using the l_inf norm
@param p: tuple of size 2 containing the coordinates of the pixel in the image
@param dist: the distance at which we're looking for neighbors of pixel p (the used distance is the l_inf norm) so all the neighbors are
in a square of side 2*dist and with center the pixel p
@param lx: the length of the image
@param ly: the width of the image
@return: a set containing the neighbors of pixel p at distance dist
""" 
def get_neighbors_linf_norm(p,dist,lx,ly):
    x=p[0]
    y=p[1]
    # c1=(x+dist,y+dist)
    # c2=(x-dist,y+dist)
    # c3=(x-dist,y-dist)
    # c4=(x+dist,y-dist)
    neighbors=[]
    if (y+dist <ly):
        for i in range(max(x-dist,0),min(lx,x+dist+1)):
                neighbors.append((i,y+dist))       
    if (y-dist >=0):
        for i in range(max(x-dist,0),min(lx,x+dist+1)):
            neighbors.append((i,y-dist))
    if (x-dist>=0):
        for j in range(max(0,y-dist),min(ly,y+dist+1)):
            neighbors.append((x-dist,j))      
    if (x+dist<lx):
        for j in range(max(0,y-dist),min(ly,y+dist+1)):
            neighbors.append((x+dist,j))      
    return set(neighbors)
"""
Gives the index of each pixel in the color histogram. This is used to identify the color of each pixel
@param X: 3D-array representing the image
@param bins: a list of size 3 containing at each position the number of parts to which each color channel is divide
@param lim_colors: a list of size 3 that gives the limite value for each color channel (depends on the color space)
@return 2D-array identifying at each position the color index of the pixel in the color histogram
"""
def getColors(X,bins,lim_colors):
    lx=X.shape[0]
    ly=X.shape[1]
    colors_hist=np.zeros((lx,ly))
    for i in range(lx):
        for j in range(ly):
            q1=X[i,j,0]//int(lim_colors[0]/bins[0])
            q2=X[i,j,1]//int(lim_colors[1]/bins[1])
            q3=X[i,j,2]//int(lim_colors[2]/bins[2])
            colors_hist[i,j]=q1*(bins[1]*bins[2])+q2*bins[2]+q3
    return colors_hist.astype(int)
"""
Computes the color correlogram. First we identify the color index of each pixel in a color histogram defined by color_space and bins, so that we obtain a quantization of colors in the image. Then, for pixel, we determine the neighbors at certain distances and check if they have the same color or not (as defined by the color histogram quantization). Finally, we normalize the obtained array.
@param X: 3D-array representing the image
@param color_space: the color space (rgb or hsv)
@param bins: a list of length 3 (is used to compute the color histogram)
@param distances: the list of distances at which we're looking for neighbors
"""
def color_correlogram(X,color_space,bins,distances):
    lx=X.shape[0]
    ly=X.shape[1]
    if color_space=='rgb':
        lim_colors=[256,256,256]
    elif color_space=='hsv':
        lim_colors=[180,256,256]
        X=cv.cvtColor(X, cv.COLOR_BGR2HSV)
    n_colors=bins[0]*bins[1]*bins[2]
    correlogram=np.zeros((len(distances),n_colors))
    colors_hist=getColors(X,bins,lim_colors)
    ki=0
    for k in distances:
        count_color=0
        for i in range(lx):
            for j in range(ly):
                ci=colors_hist[i,j]
                neighbors=get_neighbors_linf_norm((i,j),k,lx,ly)
                for p in neighbors:
                    if ci==colors_hist[p[0],p[1]]:
                        count_color+=1
                        correlogram[ki,ci]+=1
        correlogram[ki,:]=correlogram[ki,:]/count_color
        ki+=1
    return correlogram.flatten()
