import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from Visualisation import read_image
from color_histogram import color_histogram_compute


def linf_norm(p1,p2):
    # A pixel is designed by its position on the image
    # Type= tuple (x,y)
    return max(abs(p1[0]-p2[0]),abs(p1[1]-p2[1]))
    
    
# def get_neighbors(X,p,dist):
#     lx=X.shape[0]
#     ly=X.shape[1]
#     p_neigh=[]
#     for i in range(lx):
#         for j in range(ly):
#             if linf_norm(p,(i,j))==dist :
#                 p_neigh.append((i,j))
#     return p_neigh


def get_neighbors_linf_norm(p,dist,lx,ly):
    x=p[0]
    y=p[1]
    # c1=(x+dist,y+dist)
    # c2=(x-dist,y+dist)
    # c3=(x-dist,y-dist)
    # c4=(x+dist,y-dist)
    
    neighbors=[]
    
    if (y+dist <ly):
        for i in range(max(x-dist,0),min(lx,x+dist+1)):
                neighbors.append((i,y+dist))
            
    if (y-dist >=0):
        for i in range(max(x-dist,0),min(lx,x+dist+1)):
            neighbors.append((i,y-dist))
     
    if (x-dist>=0):
        for j in range(max(0,y-dist),min(ly,y+dist+1)):
            neighbors.append((x-dist,j))
            
    if (x+dist<lx):
        for j in range(max(0,y-dist),min(ly,y+dist+1)):
            neighbors.append((x+dist,j))
            
    return set(neighbors)
    
    

def dominant_colors(X,color_space,n_colors):
    
    if color_space=='hsv':
        X=cv.cvtColor(X, cv.COLOR_BGR2HSV)
        
    X=np.reshape(X,(X.shape[0]*X.shape[1],3))
    X = np.float32(X)
    flags = cv.KMEANS_RANDOM_CENTERS
    # flags = cv.KMEANS_PP_CENTERS
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,labels,centers=cv.kmeans(X,n_colors,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    
    return labels
    
def reassign_labels(labels):
    count=np.zeros(max(labels[:,0])+1)
    for i in range(labels.shape[0]):
        count[labels[i][0]]+=1
    labels_sorted=np.argsort(count)[::-1]
    new_labels=np.zeros((labels.shape[0],1))
    
    for l in labels_sorted:
        index=np.where(labels[:,0]==l)
        new_labels[index,0]=int(l)
    return new_labels.astype(int)
    
    
    
    
def color_correlogram(X,color_space,n_colors,distances):
    lx=X.shape[0]
    ly=X.shape[1]
    correlogram=np.zeros((len(distances),n_colors))
    # dominant_colors=get_dominant_colors(X,color_space,bins,n_colors)
    labels=dominant_colors(X,color_space,n_colors)
    labels=reassign_labels(labels)
    labels=np.reshape(labels,(lx,ly))
    
    ki=-1
    for k in distances:
        ki+=1
        count_color=0
        for i in range(lx):
            for j in range(ly):
                neighbors=get_neighbors_linf_norm((i,j),k,lx,ly)
                for p in neighbors:
                    
                    if labels[p[0],p[1]]==labels[i,j]:
                        correlogram[ki,labels[i,j]]+=1
                        count_color+=1
                        # if ki==0:
                        #     print(i,j)
        correlogram[ki,:]=correlogram[ki,:]/count_color
        # correlogram[ki,:]=correlogram[ki,:]/sum(correlogram[ki,:])
    return correlogram

def color_correlogram_analysis(n1,n2,color_space,n_colors,distances):
    X_all=np.empty((0,n_colors*len(distances)))
    for i in range(n1,n2):
        for j in range(3):
            print(i,j)
            X=read_image(i,j)
            corr= color_correlogram(X,color_space,n_colors,distances)
            X_all=np.vstack((X_all,corr.reshape(-1)))
    return X_all