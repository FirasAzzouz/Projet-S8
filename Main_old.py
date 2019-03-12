
from color_histogram import color_analysis
from color_moments import moments_features
from color_correlogram import color_correlogram_analysis
from shape_descriptor import fft_shape_analysis
from Local_Binary_pattern import lbp_analysis

from Clustering import clustering_kmeans


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA


n1=300
n2=320
n_clusters_=n2-n1
color_space='hsv'
bins=[4,4,4]
n_colors=8
distances=[1,4,9]
n_fft=64

classes_true=[]
for i in range(0,n2-n1):
    classes_true=classes_true+[i]*3
classes_true=np.array(classes_true)

X_all=np.empty(((n2-n1)*3,0))
# 
# X_add=color_analysis(n1,n2,color_space,bins)
# X_all=np.hstack((X_all,X_add))
# 
X_add=moments_features(n1,n2,color_space)
X_all=np.hstack((X_all,X_add))
# 
# # X_add=color_correlogram_analysis(n1,n2,color_space,n_colors,distances)
# # X_all=np.hstack((X_all,X_add))
# 
# X_add=fft_shape_analysis(n1,n2,n_fft)
# X_all=np.hstack((X_all,X_add))

# X_add=lbp_analysis(n1,n2,1e3)
# X_all=np.hstack((X_all,X_add))

classes=clustering_kmeans(X_all,n_clusters_)


print("**************Clustering_results*********************")
print('adjusted_rand_score=%0.2f' % metrics.adjusted_rand_score(classes_true,classes))
# print('normalized_mutual_info_score=%0.2f' % metrics.normalized_mutual_info_score(classes_true,classes))
print('homogeneity_score=%0.2f' % metrics.homogeneity_score(classes_true,classes))
print('completeness_score=%0.2f' % metrics.completeness_score(classes_true,classes))
print('v_measure_score=%0.2f' % metrics.v_measure_score(classes_true,classes))
print('fowlkes_mallows_score=%0.2f' % metrics.fowlkes_mallows_score(classes_true,classes))
print("\n")

pca = PCA(n_components=2,svd_solver='auto')
X_pca  =pca.fit_transform(X_all)


print("**************Clustering_results_after_pca*********************")
print('adjusted_rand_score=%0.2f' % metrics.adjusted_rand_score(classes_true,classes_pca))
# print('normalized_mutual_info_score=%0.2f' % metrics.normalized_mutual_info_score(classes_true,classes_pca))
print('homogeneity_score=%0.2f' % metrics.homogeneity_score(classes_true,classes_pca))
print('completeness_score=%0.2f' % metrics.completeness_score(classes_true,classes_pca))
print('v_measure_score=%0.2f' % metrics.v_measure_score(classes_true,classes_pca))
print('fowlkes_mallows_score=%0.2f' % metrics.fowlkes_mallows_score(classes_true,classes_pca))

classes_pca=clustering_kmeans(X_pca,n_clusters_)
plt.figure()
plt.scatter(X_pca[:,0],X_pca[:,1],c=classes_true)
plt.figure()
plt.scatter(X_pca[:,0],X_pca[:,1],c=classes_pca)
plt.show()
