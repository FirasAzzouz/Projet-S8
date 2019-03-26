@author Diallo Alassane

import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import skimage.feature as ft
from skimage import data,io
import cv2 as cv


"""
Our objectif here consists on defining functions in order 
to find features of images by using one of the technique 
of feature extraction call Local Binary Pattern . 

"""


# get_ipython().run_line_magic('matplotlib', 'inline')

# settings for LBP
METHOD = 'uniform'
P = 16
R = 2
matplotlib.rcParams['font.size'] = 9


"""
this function takes as parameters a given 
(@img : image ) and another value (@ hist_size)  representing 
the size of the histogram we want and returns the histogram of the Local Binary pattern .
Inside our function , we use a predefined function called local_binary_pattern of skimage 
which determines the local binary pattern of our image 

"""
def lbp(img,hist_size):
    img=rgb2gray(img)
    lbp=ft.local_binary_pattern(img, P, R, METHOD)
    a=np.min(lbp)
    b=np.max(lbp)
    lbp_hist,b = np.histogram(lbp.ravel(),hist_size,[a,b])
    return lbp_hist.ravel()
    # return lbp


"""
KullBack_leibler_divergence determines the information lost of 
a distribution p approximated to a distribution q 
That can thus helps us determine later the score of our algorithm 
"""
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))




"""
Return the best image corresponding to the Local Binary pattern of an image 
take an image and detrmines the histogram of its Local binary pattern . 
Then compares this histogram to histogram of a list of images given by refs. 

@img : the input image 
@refs : list of images to compare our image with 


"""
def match(refs, img):
    best_score = 10
    best_name = None
    lbp = ft.local_binary_pattern(img, P, R, METHOD)
    hist, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, normed=True, bins=P + 2,range=(0, P + 2))
    score = kullback_leibler_divergence(hist, ref_hist)
    if score < best_score:
        best_score = score
        best_name = name
    return best_name

"""
Transforms our image to a gray image 
@rgb : input image 
"""
def rgb2gray(rgb):
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray
