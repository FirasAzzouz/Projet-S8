import numpy as np
import cv2 as cv
from matplotlib.image import imread
import matplotlib.pyplot as plt

from Image_reading import *


def plot_image(image):
    plt.imshow(image)
    plt.show()
    
   
def plot_collection(n1,n2):
    fig,ax=plt.subplots(n2-n1,3)
    
    
    ax[0,0].set_title("center")
    ax[0,1].set_title("left")
    ax[0,2].set_title("right")
    for i in range(n1,n2):
        for j in range(3):
            image=read_image(i,j)
            ax[i-n1,j].imshow(image)
            ax[i-n1,j].axis('off')
            
    plt.show()




