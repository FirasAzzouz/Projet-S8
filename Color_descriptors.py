import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import cv2 as cv

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
