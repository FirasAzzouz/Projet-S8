# author: Azzouz
# -*- coding: utf-8 -*-

"""
This file contains functions to read images from datasets aloi_red4_stereo and aloi_red4_view.
"""
import cv2 as cv
from matplotlib.image import imread
import os.path

dataset_path_1=os.path.join(os.path.abspath("datasets"),"aloi_red4_stereo")
dataset_path_2=os.path.join(os.path.abspath("datasets"),"aloi_red4_view")

"""
Reads an image from the dataset aloi_red4_stereo
@param i : number of the object (from 251 to 1000)
@param j: number of the photo for object i (from 0 to 2)
@return 3-D array (144*192*3): matrix representation of an image in an RGB color space
"""
def read_image_1(i,j):
    Image_name=[str(i)+'_c.png',str(i)+'_l.png',str(i)+'_r.png']
    Image_path=os.path.join(dataset_path_1,str(i)+'\\',Image_name[j])
#     print (Image_path)
    image = cv.imread(Image_path)
    if image is None:
        print("Check the path")
    b,g,r = cv.split(image)           
    image = cv.merge([r,g,b])
#     image=plt.imread(Image_path)
    return image

"""
Reads an image from the dataset aloi_red4_view
@param i : number of the object (from 1 to 100)
@param j: number of the photo for object i (from 0 to 71)
@return 3-D array (144*192*3): matrix representation of an image in an RGB color space
"""
def read_image_2(i,j):
    Image_name=str(i)+'_r'+str(j*5)+'.png'
    Image_path=os.path.join(dataset_path_2,str(i)+'\\',Image_name)
#     print (Image_path)
    image = cv.imread(Image_path)
    if image is None:
        print("Check the path")
    b,g,r = cv.split(image)           
    image = cv.merge([r,g,b])
#     image=plt.imread(Image_path)
    return image