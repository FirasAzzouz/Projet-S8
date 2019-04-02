import random
import math
import numpy as np

# ******************************************KMeans Clustering with sklearn***************************************
from sklearn.cluster import KMeans
def kmeans_clustering_sklearn(X,n_clusters_):
    k_means = KMeans(init='k-means++', n_clusters=n_clusters_)
    k_means.fit(X_norm)
    classes=k_means.labels_
    return classes

# *********************************************************************************************************************
# ******************************Implementation of KMeans with several distance definitions*****************************
# *********************************************************************************************************************


# ************************************************Distances definition*************************************************
#Euclidean Distance between two d-dimensional points
def eucl_dist(a, b, axis):
    return np.linalg.norm(a - b, axis=axis)
                          
def distance(a,b,dist_type,axis=1):
    if dist_type=="euclidean":
        return eucl_dist(a,b,axis)
    else:
        print("Distance defintion not found")


# ***************************************************KMeans algorithm**************************************************    
#K-Means Algorithm
def kmeans(datapoints,k,dist_type):

    # d - Dimensionality of Datapoints
    d = len(datapoints[0]) 
    
    #Limit our iterations
    Max_Iterations = 1000
    i = 0
    
    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    
    #Randomly Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0,k):
        new_cluster = []
        #for i in range(0,d):
        #    new_cluster += [random.randint(0,10)]
        cluster_centers += [random.choice(datapoints)]
        
        
        #Sometimes The Random points are chosen poorly and so there ends up being empty clusters
        #In this particular implementation we want to force K exact clusters.
        #To take this feature off, simply take away "force_recalculation" from the while conditional.
        force_recalculation = False
    
    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :
        
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
    
        #Update Point's Cluster Alligiance
        for p in range(0,len(datapoints)):
            min_dist = float("inf")
            
            #Check min_distance against all centers
            for c in range(0,len(cluster_centers)):
                dist = distance(datapoints[p],cluster_centers[c],dist_type)
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c   # Reassign Point to new Cluster
        
        #Update Cluster's Position
        for k in range(0,len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster[p] == k): #If this point belongs to the cluster
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1
            
            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members) 
                
                #This means that our initial random assignment was poorly chosen
                #Change it to a new datapoint to actually force k clusters
                else: 
                    new_center = random.choice(datapoints)
                    force_recalculation = True
                    print ("Forced Recalculation...")

            cluster_centers[k] = new_center
    
        
#     print "======== Results ========"
#     print "Clusters", cluster_centers
    print ("Iterations",i)
#     print "Assignments", cluster
    return cluster


# ***************************************************KMeans++ algorithm**************************************************  
# Kmeans++
def get_center(X,k,dist_type):
    temp = []
    temp.append(X[np.random.randint(0, len(X))])
    while len(temp)<k:
        d2= np.array([min([np.square(distance(i,c,dist_type,None)) for c in temp]) for i in X])
        prob = d2/d2.sum()
        cum_prob = prob.cumsum()
        r = np.random.random()
        ind = np.where(cum_prob >= r)[0][0]
        temp.append(X[ind])
    return np.array(temp)

def kmeans_pp(x,k,dist_type):
    
    # initializing cluster variable
    cluster = np.zeros(x.shape[0])
    
    # initializing clusters centers
    center = get_center(x,k,dist_type)

    # assigining zeros to old centroids value
    center_old = np.zeros(center.shape)

    # initial error
    err = distance(center, center_old, dist_type, None)

    while err != 0:

        # calculatin distance of data points from centroids and assiging min distance cluster centroid as data point cluster
        for i in range(len(x)):
            distances = distance(x[i], center,dist_type)
            clust = np.argmin(distances)
            cluster[i] = clust

        # changing old centroids value
        center_old = np.copy(center)

        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [x[j] for j in range(len(x)) if cluster[j] == i]
            if points:
                center[i] = np.mean(points, axis=0)

        # calculation difference between new centroid and old centroid values
        err = distance(center, center_old, dist_type, None)

    # calculation total difference between cluster centroids and cluster data points
    error = 0
    for i in range(k):
        d = [distance(x[j], center[i], dist_type, None) for j in range(len(x)) if cluster[j] == i]
        error += np.sum(d)

    # counting data points in all clusters
    count = {key: 0.0 for key in range(k)}
    for i in range(len(x)):
        count[cluster[i]] += 1

    # displaying cluster number, average distance between centroids and data points and cluster count
    print (k, error / len(x), count)

    return cluster
