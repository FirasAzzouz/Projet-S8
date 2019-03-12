import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from Image_reading import read_image



    
def color_analysis(n1,n2,color_space,bins):
    
    X_all=np.empty((0,int(bins[0]*bins[1]*bins[2])))
    for i in range(n1,n2):
        for j in range(3):
            # print(i,j)
            X=read_image(i,j)
            hist= color_histogram(X,color_space,bins)
            hist=histogram_filter(hist)
            hist=histogram_normalization(hist)
            
            # plt.figure()
            # plt.bar(np.arange(int(bins[0]*bins[1]*bins[2])),hist.flatten())
            
            X_all=np.vstack((X_all,hist))
    # plt.show()
    return X_all

def histogram_normalization(hist):
    hist_norm=hist/sum(hist)
    return hist_norm

def histogram_filter(hist):
    for i in range(len(hist)):
        if hist[i]>500:
            hist[i]=0
    return hist



def color_histogram(X,color_space,bins):
    """ 
	This function computes the color histogram of an image X in a color space using the specified bins.
	@param: X: 2D-Array: representation of an image in a color space
	@param: color_space: String : color space, typically "rgb" or "hsv"
	@param: bins: list of 3 integers: the chosen bins to define the colors. Each value represents the number of intervals
	to which each color channel is divided.
	@returns: 1-D Array: the color histogram 
    """

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

def color_histogram_hsv(X,bins):
    """
    bins : list
    
    """
    lim_h=180
    lim_s=256
    lim_v=256
    X=cv.cvtColor(X, cv.COLOR_BGR2HSV)
    enum=np.zeros((int(bins[0]*bins[1]*bins[2]),))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            q1=int(X[i,j,0])//int(lim_h/bins[0])
            q2=int(X[i,j,1])//int(lim_s/bins[1])
            q3=int(X[i,j,2])//int(lim_v/bins[2])
            enum[int(q1*(bins[1]*bins[2])+q2*bins[2]+q3)]+=1
    # plt.bar(np.arange(int(bins[0]*bins[1]*bins[2])),enum)
    # plt.show()
    return enum
    
    

def color_histogram_rgb(X,bins):
    lim_colors=256
    # n_bins=4
    enum=np.zeros((int(bins[0]*bins[1]*bins[2]),))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # print(i,j)
            # q1=int(X[i,j,0]*lim_colors)//int(lim_colors/n_bins)
            # q2=int(X[i,j,1]*lim_colors)//int(lim_colors/n_bins)
            # q3=int(X[i,j,2]*lim_colors)//int(lim_colors/n_bins)
            q1=int(X[i,j,0])//int(lim_colors/bins[0])
            q2=int(X[i,j,1])//int(lim_colors/bins[1])
            q3=int(X[i,j,2])//int(lim_colors/bins[2])
            # if q1==n_bins:
            #     q1=n_bins-1
            # if q2==n_bins:
            #     q2=n_bins-1
            # if q3==n_bins:
            #     q3=n_bins-1
            enum[int(q1*(bins[1]*bins[2])+q2*bins[2]+q3)]+=1
    # plt.bar(np.arange(n_bins**3),enum)
    # plt.show()
    return enum
    

    