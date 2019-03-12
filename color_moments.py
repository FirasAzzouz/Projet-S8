import numpy as np
import matplotlib.pyplot as plt
from Image_reading import read_image
from scipy.stats import skew, kurtosis
import cv2 as cv

def moments_features(n1,n2,color_space):
    X_all=np.empty((0,15))
    for i in range(n1,n2):
        for j in range(3):
            # print(i,j)
            X=read_image(i,j)
            ftr=moments_calcul(X)
            X_all=np.vstack((X_all,ftr))
    return X_all
    
    
def moments_calcul(X,color_space):
    if color_space=='hsv':
        X=cv.cvtColor(X, cv.COLOR_BGR2HSV)
    elif color_space=='rgb':
        continue
    else:
        print("select please a color space among rgb and hsv")
    X0=X[:,:,0].reshape(-1)
    X1=X[:,:,1].reshape(-1)
    X2=X[:,:,2].reshape(-1)
    Mean_=np.array([np.mean(X0),np.mean(X1),np.mean(X2)])
    Var_=np.array([np.var(X0),np.var(X1),np.var(X2)])
    Str_=np.array([np.std(X0),np.std(X1),np.std(X2)])
    Skewness_=np.array([skew(X0),skew(X1),skew(X2)])
    Kurtosis_=np.array([kurtosis(X0),kurtosis(X1),kurtosis(X2)])
    
    return np.hstack((Mean_,Var_,Str_,Skewness_,Kurtosis_))



