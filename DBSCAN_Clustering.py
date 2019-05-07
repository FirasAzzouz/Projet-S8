# author: Azzouz
# -*- coding: utf-8 -*-

# ******************************************KMeans Clustering with sklearn***************************************
from sklearn.cluster import DBSCAN
def dbscan_clustering_sklearn(X,eps_,minPts_,metric_):
    dbscan = DBSCAN(eps=eps_, min_samples=minPts_,metric=metric_).fit(X)
    dbscan.fit(X)
    classes=dbscan.labels_
    return classes