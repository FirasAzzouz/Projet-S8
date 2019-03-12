import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from Visualisation import read_image
from color_histogram import color_histogram_compute


def linf_norm(p1,p2):
    # A pixel is designed by its position on the image
    # Type= tuple (x,y)
    return max(abs(p1[0]-p2[0]),abs(p1[1]-p2[1]))
    
def same_color(X,bins,p1,p2):
    
    c1=X[p1[0],p1[1]]
    c2=X[p2[0],p2[1]]
    
    q11=c1[0]//bins[0]
    q12=c1[1]//bins[1]
    q13=c1[2]//bins[2]
    
    q21=c2[0]//bins[0]
    q22=c2[1]//bins[1]
    q23=c2[2]//bins[2]
    
    return (q11,q12,q13)==(q21,q22,q23)
    
    
    
    
def get_neighbors(X,p,dist):
    lx=X.shape[0]
    ly=X.shape[1]
    p_neigh=[]
    for i in range(lx):
        for j in range(ly):
            if linf_norm(p,(i,j))==dist :
                p_neigh.append((i,j))
    return p_neigh
    
# def get_dominant_colors(X,color_space,bins,n):
#     hist=color_histogram_compute(X,color_space,bins)
#     dominant_colors=[]
#     for k in range(n_colors):
#         c=np.argmax(hist)
#         dominant_colors.append(c)
#         hist[c]=0
#     return dominant_colors
    
def isColor(pixel,c,bins):
    q1=pixel[0]//bins[0]
    q2=pixel[1]//bins[1]
    q3=pixel[2]//bins[2]
    
    c_p=q1*(bins[1]*bins[2])+q2*bins[2]+q3
    return c_p==c
    
    
def color_correlogram(X,color_space,bins,distances):
    lx=X.shape[0]
    ly=X.shape[1]
    correlogram=np.zeros((len(distances),n_colors))
    # dominant_colors=get_dominant_colors(X,color_space,bins,n_colors)
    n_colors=bins[0]*bins[1]*bins[2]
    dominant_colors=np.arange(n_colors)
    i_k=-1
    for k in distances:
        i_k=+1
        count_color=0
        i_c=-1
        for c in dominant_colors:
            i_c=+1
            for i in range(lx):
                for j in range(ly):
                    if isColor(X[i,j],c,bins):
                        neighbors=get_neighbors(X,(i,j),k)
                        for p in neighbors:
                            count_color=+1
                            if same_color(X,bins,(i,j),p):
                                correlogram[i_k,i_c]=+1
                        
        correlogram[i_k,:]=correlogram[i_k,:]/count_color
    return correlogram
    