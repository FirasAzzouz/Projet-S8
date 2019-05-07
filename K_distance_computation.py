import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

def K_distance(X, k, metric_):
    """
        compute the k-distance for each point
        
        :param X: the input matrix containing the information
        :param k: number of neighbors (= minPts in this specific case)
        :param metric_: the distance definition
        
        :type X: 2-D array
        :type k: int
        :type metric_: String
       
        :return: Knd : K-distances 
        :rtype: 1-D array
    """
  
    # Distance matrix computation
    Mat_dist = pairwise_distances(X, metric = metric_)
            
    # K-Distance computation
    K_minima_mat = find_minima(Mat_dist, k)
    Knd = np.mean(K_minima_mat, 1)
    
    # K-Distance sort
    Knd = np.sort(Knd)[::-1]
    
    return Knd
            
def find_minima(Mat_dist,k):
    """
        find the distances to the k nearest neighbors
        
        :param Mat_dist: distance matrix
        :param k: number of neighbors = minPts
        
        :type Mat_dist: 2-D array 
        :type k: int
       
        :return: K_minima_mat : matrix containing at each line the distances to the k nearest neighbors
        :rtype: K_minima_mat : 2-D array
    """
    K_minima_mat = np.zeros((Mat_dist.shape[0], k))
    for i in range(Mat_dist.shape[0]):
        pp = np.sort(Mat_dist[i])
        K_minima_mat[i] = pp[1:k+1]
    return K_minima_mat
