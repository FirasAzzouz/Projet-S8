
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from Image_reading import *
from Texture_descriptors import lbp 


if __name__ == '__main__':
#   Parameters defintion:
    n1=1 # First image (included)
    n2=21 # Last image (excluded)
    n_img=72 # number of images per object
    n_clusters_=n2-n1 # number of objects
    hist_size = 12 # size of the histogram for the lbp descriptor
    
#     Classes_true is a vector containing the true classes
    classes_true=[]
    for i in range(0,n2-n1):
        classes_true=classes_true+[i]*n_img
    classes_true=np.array(classes_true)

#    We regroup the images features in a single 2D-Array where each line is the feature vector of the image
    X_all=[]
    for i in range(n_img*(n2-n1)):
        X_all.append([])
    for i in range(n2-n1):
        for j in range(n_img):
            X=read_image_2(i+n1,j)
            # ********************Texture_features***********************
            lbp_ftr = lbp(X,hist_size)
            X_all[n_img*i+j]=X_all[n_img*i+j]+[lbp_ftr]
            X_all[n_img*i+j]=np.concatenate(X_all[n_img*i+j])
    
    X_all=np.array(X_all).astype(float)
    print("Number of images= "+ str(X_all.shape[0]))
    print("Number of features= "+ str(X_all.shape[1]))
    print("\n")
    
    
    # *********************Visualisation**********************
    X_norm = StandardScaler().fit_transform(X_all)
    X_embedded = TSNE(n_components=2).fit_transform(X_norm)
    
    plt.figure()
    plt.scatter(X_embedded[:,0],X_embedded[:,1],c=classes_true)
    plt.title("Visualization of the true classes (only lbp features)")
    plt.show()


# Apres l'extraction des features(12 au total) des images de notre base de données aloi-red4-view , la visualisation du comportement des images (au nombre de 1440)dans la base de donnée se represente par le dessin en dessus. L'analyse qui en decoulerait serait que les images de certains object sont bien regroupées . Toute fois , il en est pas de même des objects qu'on visualise sur le centre du dessin . 

# Cela pourrait s'expliquer par le modele du Local binary pattern qui dans son execution tient compte de certains criteres qui font que le resultat final se presente comme sur la visualisation
# 
# En Somme , il nous seras difficile de discerner les clusters en utilisant le model local Binary pattern . En revanche on pourrait l'utiliser comme modele supplementaire d'extraction de features pouvant eventuellement marcher sur d'autre base de donnée . 
