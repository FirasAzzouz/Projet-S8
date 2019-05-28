import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import cv2 as cv

from Image_reading import *
from Color_descriptors import *
from Shape_descriptors import *
from Texture_descriptors import lbp

from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabaz_score, davies_bouldin_score
from sklearn.manifold import TSNE

if __name__ == '__main__':
    
    n1 = 251
    n2 = 401
    n_clusters_ = n2-n1
    color_space = 'hsv'
    bins = [4,4,4]
    n_colors = 8
    n_fft = 64
    hist_size_lbp = 100
    
    pos = ['c'] # c, l, ou r
    
    X_all=[]
    imgs = []
    for i in range(len(pos)*(n2-n1)):
        X_all.append([])
    
    for i in range(n2-n1):
        for j in range(len(pos)):
            
            filename = 'datasets/aloi_red4_stereo/%s/%s_%s.png' %(n1+i,n1+i,pos[j])
            
            X = cv.imread(filename)
            X = cv.cvtColor(X, cv.COLOR_BGR2RGB)
            imgs.append(X)
            
            # ********************Color_features***********************
            moments_ftr=moments_calcul(X,color_space)
            X_all[len(pos)*i+j]=X_all[len(pos)*i+j]+[moments_ftr]
            
            hist= color_histogram(X,color_space,bins)
            # hist=histogram_filter(hist)
            # hist=histogram_normalization(hist)
            X_all[len(pos)*i+j]=X_all[len(pos)*i+j]+[hist]
            
            
            
            # ********************Shape_features***********************
            shape_fft=concatShapeFeatures(X,n_fft)
            X_all[len(pos)*i+j]=X_all[len(pos)*i+j]+[shape_fft]
            
            
            # ********************Texture_features***********************
            lbp_feature=lbp(X,hist_size_lbp)
            X_all[len(pos)*i+j]=X_all[len(pos)*i+j]+[lbp_feature]
            
            
            
            X_all[len(pos)*i+j]=np.concatenate(X_all[len(pos)*i+j])
            
            #X_all[len(pos)*i+j] = X.flatten()
            
    # réduction de dimension
    X_all=np.array(X_all).astype(float)
    dim_pca = min(X_all.shape[1],X_all.shape[0])
    pca = PCA(n_components = dim_pca)
    pca.fit(X_all)
    
    dim = 2
    for i in np.logspace(1, np.log2(dim_pca-1), 15,base =2):
        if np.sum(pca.explained_variance_ratio_[:np.int64(np.floor(i))]) > 0.85:
            dim = np.int64(np.floor(i))
            break
    
    pca = PCA(n_components = dim)
    X_all = pca.fit_transform(X_all)  
    print("Number of images= "+ str(X_all.shape[0]))
    print("Number of features= "+ str(X_all.shape[1]))
    print("\n")
    
    # normalisation
    X_norm = StandardScaler().fit_transform(X_all.astype(float))
    X_embedded = TSNE(n_components=2, perplexity = 10).fit_transform(X_norm)
    
    # dbscan 
    meanshift = MeanShift(bandwidth = 2.8).fit(X_norm)
    
    k = np.unique(meanshift.labels_).shape[0]
    print('Number of clusters=', k)
    
    # visualiser les clusters
    for cluster in range(min(k,5)):
        idx = np.where(meanshift.labels_ == cluster)[0]
        n_imgs_ci = len(idx)
        dim = int(np.ceil(n_imgs_ci**0.5))
        if dim == 1:
            dim += 1
        fig, ax = plt.subplots(dim,dim)
        
        for i in range(n_imgs_ci):
            ax[i//dim,i%dim].imshow(imgs[idx[i]])
            ax[i//dim,i%dim].set_xticks([])
            ax[i//dim,i%dim].set_yticks([])
        for i in range(n_imgs_ci,dim**2):
            fig.delaxes(ax.flatten()[i])
    
    
    
    # visualisation avec t-SNE
    
    plt.figure()
    plt.scatter(X_embedded[:,0],X_embedded[:,1],c=meanshift.labels_)
    plt.title("Visualisation of the true classes")
    plt.show()

    # evaluation by internal indices
    
    print("Silhouette score =" , silhouette_score(X_norm,meanshift.labels_))
    # between -1 and 1, score of 1 implies good clustering, close to 0 means 
    # overlapping clusters
    print("Calinski-Harabaz score =" , calinski_harabaz_score(X_norm,meanshift.labels_))
    # higher score implies good separation between clusters
    #print("Davies Bouldin score =" , davies_bouldin_score(X_norm,dbscan.labels_))
    # always positive, closer value to 0 indicates better separation
    