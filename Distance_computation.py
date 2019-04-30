# author: Azzouz
# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial.distance as dist 

# ************************************************Distances definition*************************************************
 
def chord(a,b):
    return np.linalg.norm(a*(1*np.linalg.norm(a))-b*(1*np.linalg.norm(b)))
    
def distance(a,b,dist_type):
    if dist_type=="euclidean":
        return dist.euclidean(a,b)
    elif dist_type=="manhattan":
        return dist.cityblock(a,b)
    elif dist_type=="chebyshev":
        return dist.chebyshev(a,b)
    elif dist_type=="cosine":
        return dist.cosine(a,b)
    elif dist_type=="correlation":
        return dist.correlation(a,b)
    elif dist_type=="braycurtis":
        return dist.braycurtis(a,b)
    elif dist_type=="canberra":
        return dist.canberra(a,b)
    elif dist_type=="chord":
        return dist.chord(a,b)
    else:
        print("Distance defintion not found")