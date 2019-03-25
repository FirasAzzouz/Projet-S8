# author: Azzouz
# -*- coding: utf-8 -*-
"""
This file is used to plot images from the datasets separately or in a collection
"""
import numpy as np
import cv2 as cv
from matplotlib.image import imread
import matplotlib.pyplot as plt

from Image_reading import *

"""
Shows an image
@ param image: 3D-Array representing an image in an RGB color space
"""
def plot_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
"""
Plots a collection of images from n1(included) to n2(excluded) of the dataset aloi_red4_stereo (3 images per object)
@param n1: first object (from 251 to 1000)
@param n2: last object (from n1+1 to 1001)
"""
def plot_collection_1(n1,n2):
    if (n2>n1) :
        fig,ax=plt.subplots(n2-n1,3,figsize=(10,10))
        
        ax[0,0].set_title("center")
        ax[0,1].set_title("left")
        ax[0,2].set_title("right")
        for i in range(n1,n2):
            for j in range(3):
                image=read_image_1(i,j)
                ax[i-n1,j].imshow(image)
                ax[i-n1,j].axis('off')
        plt.subplots_adjust(wspace=0.05)
        plt.show()
    else:
        print("Incorrect choice of parameters")
"""
Plots the image of a single object in the dataset aloi_red4_view (72 images per object)
@param i: the number of the object
"""
def plot_object_2(i):
    b=6
    fig,ax=plt.subplots(72//b,b,figsize=(10,10))
    for j in range(72):
        image=read_image_2(i,j)
        j1=j//b
        j2=j%b
        ax[j1,j2].imshow(image)
        ax[j1,j2].axis('off')
    plt.subplots_adjust(wspace=0.05)
    plt.show()
    
        
    




