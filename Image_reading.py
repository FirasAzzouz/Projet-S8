import cv2 as cv
from matplotlib.image import imread
import os.path

dataset_path=os.path.join(os.path.abspath("datasets"),"Objects_AmsLib")


def read_image(i,j):
    Image_name=[str(i)+'_c.png',str(i)+'_l.png',str(i)+'_r.png']
    Image_path=os.path.join(dataset_path,str(i)+'\\',Image_name[j])
#     print (Image_path)
    image = cv.imread(Image_path)
#     image=plt.imread(Image_path)
    if image is None:
        print("Check the path")
    return image