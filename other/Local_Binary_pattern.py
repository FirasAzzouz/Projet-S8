
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import skimage.feature as ft
from skimage import data,io
import cv2 as cv
from Visualisation import read_image
# get_ipython().run_line_magic('matplotlib', 'inline')

# settings for LBP
METHOD = 'uniform'
P = 16
R = 2
matplotlib.rcParams['font.size'] = 9


def lbp_analysis(n1,n2,hist_size):
    X_all=[]
    for i in range(n1,n2):
        for j in range(3):
            print(i,j)
            X=read_image(i,j)
            lbp_features=lbp(X,hist_size)
            X_all.append(lbp_features)
    return np.array(X_all)

def lbp(img,hist_size):
    img=rgb2gray(img)
    lbp=ft.local_binary_pattern(img, P, R, METHOD)
    a=np.min(lbp)
    b=np.max(lbp)
    lbp_hist,b = np.histogram(lbp.ravel(),hist_size,[a,b])
    return lbp_hist.ravel()
    # return lbp


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


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





def rgb2gray(rgb):
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray

"""
#img = io.imread('9540474.1.jpg')
img1 = rgb2gray(io.imread('9540474.1.jpg'));
img2= rgb2gray(io.imread('9540504.1.jpg'))
img3 =rgb2gray(io.imread('9540512.1.jpg'))  

refs = {
    'img1': ft.local_binary_pattern(img1, P, R, METHOD),
    'img2': ft.local_binary_pattern(img2, P, R, METHOD),
    'img3':ft.local_binary_pattern(img3, P, R, METHOD),
       }


print (match(refs, nd.rotate(img, angle=30, reshape=False)))

#  plot histograms of LBP of textures
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,figsize=(9, 6))

# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,figsize=(9, 6))
plt.gray()

ax1.imshow(img1)
ax1.axis('off')
ax4.hist(refs['img1'].ravel(), normed=True, bins=P + 2, range=(0, P + 2))
ax4.set_ylabel('Percentage')

ax2.imshow(img2)
ax2.axis('off')
ax5.hist(refs['img2']).ravel(), normed=True, bins=P + 2, range=(0, P + 2))
ax5.set_xlabel('Uniform LBP values')

ax3.imshow(img3)
ax3.axis('off')
ax6.hist(refs['img3'].ravel(), normed=True, bins=P + 2, range=(0, P + 2))

plt.show()
"""

