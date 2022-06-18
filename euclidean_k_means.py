import time
from copy import deepcopy
from math import sqrt

import numpy as np
import torch

from utils import *

'''
traditional k-means clustering

inputs: X (n rows of m-dimensional points),
        classes (n-dimensional vector)
        k (scalar)
output: classes (n-dimensional vector)
        centers (k rows of m-dimensional points)
'''
def kmeans(X, k, centers):
    start = time.time()
    X = X.to(torch.float64)

    phi = get_phi(dist_to_phi('gaussian'))
    classes = get_classes(X, centers, phi)
    points, features = X.size()

    members = torch.zeros(k)

    switched = True
    iters = 0

    #iterate until membership stabilizes
    while switched:
        start_iter = time.time()
        iters += 1

        centers_old = centers #new code added 12/27 to address bug
        centers = torch.zeros([k, features], dtype=torch.float64)
        members = torch.zeros(k, dtype=torch.float64)

        #update each cluster center. center = mean of all points assigned to it
        for p in range(points):
            i = classes[p]
            centers[i,:] += X[p,:]
            members[i] += 1

        for i in range(k):
            if members[i] >= 1:
                centers[i,:] /= members[i]
            else:
                centers[i,:] = centers_old[i,:]

        #update all the cluster assignments based on new centers
        switched = False
        for p in range(points):
            point = X[p,:]

            euclidean_dist = torch.sum((point - centers).pow(2), axis=1)
            j = torch.argmin(euclidean_dist)

            #if a cluster assignment needs to be changed, do it and record the switch
            if classes[p] != j:
                classes[p] = j
                switched = True
                
    return classes, centers, iters, time.time() - start


def power_kmeans(X, s, k, centers, y):
    start = time.time()

    X = X.to(torch.float64)
    centers = centers.to(torch.float64)
    centers_old = deepcopy(centers)
    s = float(s)

    n_points, m_features = X.size()
    weights = torch.zeros([n_points, k], dtype=torch.float64)
    epsilon = 1e-8
    iter = 0

    #for plotting
    s_0 = s
    obj_list = np.array([])
    iter_list = np.array([])

    classes_old = None
    classes = None
    convergence_cnt = 0

    while True:
        start_iter = time.time()
        iter += 1

        euclidean_dist = pairwise_distances(X, centers) #n x k distance matrix
        weights = torch.t(torch.mul(torch.t(torch.pow(euclidean_dist, (s-1))), torch.pow(torch.sum(torch.pow(euclidean_dist, s), axis=1), (1/s - 1)) + epsilon))

        if torch.min(weights) < torch.tensor([1e-280], dtype=torch.float64):
            print("weight vector small")
            break

        #update cluster centers
        centers_old = deepcopy(centers)
        centers = torch.t(torch.div(torch.t(torch.matmul(torch.t(weights), X)), torch.sum(weights, axis=0))) #m x k, k -->  m x k

        if torch.isnan(torch.norm(centers)):
            print("centers NAN'd, so terminating")
            centers = centers_old
            break

        #anneal s value
        if iter % 2 == 0:
            if s > -1.0:
                s += -.2
            elif s > -120.0:
                s *= 1.06

        if classes is not None:
            classes_old = classes

        classes = torch.argmin(pairwise_distances(X, centers), axis=1)
        if classes_old is not None and classes is not None and torch.equal(classes_old, classes):
            convergence_cnt += 1
        else:
            convergence_cnt = 0

        if convergence_cnt == 5:
            #print("point assignments have converged")
            break

    #assign cluster labels
    euclidean_dist = pairwise_distances(X, centers) #n x k distance matrix
    classes = torch.argmin(euclidean_dist, axis=1)

    #print("powermeans final s: ", s, "number of iters: ", iter)
    return classes, centers, s, iter, time.time() - start


def kmeans_obj(centers, X):
    euclidean_dist = pairwise_distances(X, centers) #n x k distance matrix
    return torch.sum(torch.min(euclidean_dist, axis=1)) #sum over n data points (1 for each pt)


def kgen_obj(centers, X, s):
    euclidean_dist = pairwise_distances(X, centers) #n x k distance matrix
    return torch.sum(powmean(torch.pow(euclidean_dist, 2), s)) #sum over n datapoints (1 for each pt)
