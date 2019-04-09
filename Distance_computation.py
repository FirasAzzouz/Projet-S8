# author: Azzouz
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import minkowski,chebyshev, cosine, correlation, braycurtis, cityblock, canberra

# ************************************************Distances definition*************************************************
 
def chord(a,b):
    return np.linalgnorm(a*(1*np.linalg.norm(a))-b*(1*np.linalg.norm(b)))
    
    
def distance(a,b,dist_type):
    if dist_type=="euclidean":
        return np.linalg.norm(a - b)
    elif dist_type=="manhattan":
        return cityblock(a,b)
    elif dist_type=="chebyshev":
        return chebyshev(a,b)
    elif dist_type=="cosine":
        return cosine(a,b)
    elif dist_type=="correlation":
        return correlation(a,b)
    elif dist_type=="braycurtis":
        return braycurtis(a,b)
    elif dist_type=="canberra":
        return canberra(a,b)
    elif dist_type=="chord":
        return chord(a,b)
    else:
        print("Distance defintion not found")