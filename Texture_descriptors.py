import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import skimage.feature as ft
from skimage import data,io
import cv2 as cv

# get_ipython().run_line_magic('matplotlib', 'inline')

# settings for LBP
METHOD = 'uniform'
P = 16
R = 2
matplotlib.rcParams['font.size'] = 9

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