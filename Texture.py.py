
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import skimage.feature as ft
from skimage import data,io
import cv2 as cv
from Image_reading import *
from Color_descriptors import *
from Texture_descriptors import lbp 


if __name__ == '__main__':
#   Parameters defintion:
    n1=1 # First image (included)
    n2=21 # Last image (excluded)
    n_img=72 # number of images per object
    n_clusters_=n2-n1 # number of objects
    color_space='hsv'
    bins=[4,4,4]
    
#     Classes_true is a vector containing the true classes
    classes_true=[]
    for i in range(0,n2-n1):
        classes_true=classes_true+[i]*n_img
    classes_true=np.array(classes_true)

#     First, we regroup the images features in a single 2D-Array where each line is the feature vector of the image
    X_all=[]
    for i in range(n_img*(n2-n1)):
        X_all.append([])
    for i in range(n2-n1):
        for j in range(n_img):
            X=read_image_1(i+n1,j)
            # ********************Texture_features***********************
            hist_size = 12
            lobina = lbp(X,hist_size)
            # hist=histogram_filter(hist)
            # hist=histogram_normalization(hist)
            X_all[n_img*i+j]=X_all[n_img*i+j]+[hist]
            X_all[n_img*i+j]=np.concatenate(X_all[n_img*i+j])
    
    X_all=np.array(X_all).astype(float)
    print("Number of images= "+ str(X_all.shape[0]))
    print("Number of features= "+ str(X_all.shape[1]))
    print("\n")
    
    
    # *********************Visualisation**********************
    

