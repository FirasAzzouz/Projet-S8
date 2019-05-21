import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances

def spectralClustering_rbf(X,n_clusters_,metric_,gamma_):
    X_dist = pairwise_distances(X, metric=metric_)
    X_dist_sqr = np.square(X_dist)
    X_affinity = np.exp(-gamma_*X_dist_sqr)
    spec = SpectralClustering(n_clusters_, affinity = "precomputed")
    spec.fit(X_affinity)
    classes = spec.labels_
    return classes