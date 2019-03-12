from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def clustering_kmeans(X,n_clusters_):
    X_norm = StandardScaler().fit_transform(X.astype(float))
    k_means = KMeans(init='k-means++', n_clusters=n_clusters_)
    k_means.fit(X_norm)
    classes=k_means.labels_
    return classes
    