import os

project_path='C:/Users/firas/OneDrive/Documents/Cours/Projet Innovation S8'
dataset_path=os.path.join(project_path,"Datasets","faces")




import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib.image import imread
import matplotlib.pyplot as plt

path_ind_1=os.path.join(dataset_path,"faces94","male")+'/'+'ahodki'
path_ind_2=os.path.join(dataset_path,"grimace")+'/'+'ste'

def plot_ind_exp(path_ind):
    
    fig,ax=plt.subplots(3,3)
    p=0
    for k in range(4,13):
        i=p//3
        j=p%3
        ax[i,j].imshow(read_image_exp(path_ind,k+1))
        ax[i,j].axis('off')
        p=p+1
    plt.show()
        
    

def read_image_exp(path_ind,j):
    i=-1
    while path_ind[i]!='/':
        i=i-1
    ind=path_ind[i+1:]
        
    Image_path=os.path.join(path_ind,ind+'_exp.'+str(j)+".jpg")
    image = cv.imread(Image_path)
    return image






def plot_ind(path_ind):
    
    fig,ax=plt.subplots(3,3)
    for k in range(0,9):
        i=k//3
        j=k%3
        ax[i,j].imshow(read_image(path_ind,k+1))
        ax[i,j].axis('off')
    plt.show()
        
    

def read_image(path_ind,j):
    i=-1
    while path_ind[i]!='/':
        i=i-1
    ind=path_ind[i+1:]
        
    Image_path=os.path.join(path_ind,ind+'.'+str(j)+".jpg")
    image = cv.imread(Image_path)
    return image
    