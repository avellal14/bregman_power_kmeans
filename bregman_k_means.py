import time
from math import sqrt
from copy import deepcopy

import numpy as np
import torch

from utils import *

def power_kmeans_bregman(phi, X, s, k, centers, n_epochs=1, lr=1e-2, iterative=False, n_iters=100, y=None, i=0, convergence_threshold=5, shape=None):
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
    failed_to_converge = False

    while True:
        start_iter = time.time()
        iter += 1

        bregman_dist = pairwise_bregman(X, centers, phi, shape)#torch.clamp(phi[0](X)[:, np.newaxis] - phi[0](centers)[np.newaxis, :] - torch.sum((X[:, np.newaxis] - centers[np.newaxis, :]) * phi[1](centers[np.newaxis, :]), axis=-1), min=1e-12, max=1e12)
        weights = torch.t(torch.mul(torch.t(torch.pow(bregman_dist, (s-1))), torch.pow(torch.sum(torch.pow(bregman_dist, s), axis=1), (1/s - 1)) + epsilon)) # n x k

        if torch.min(weights) < torch.tensor([1e-280], dtype=torch.float64):
            print("weight vector small")
            break

        #update cluster centers
        centers_old = deepcopy(centers)
        centers = torch.t(torch.div(torch.t(torch.matmul(torch.t(weights), X)), torch.sum(weights, axis=0))) #m x k, k -->  m x k

        if torch.isnan(torch.norm(centers)):
            print("centers NAN'd, so terminating")
            centers = centers_old
            failed_to_converge = True
            break

        #anneal s value
        if iter % 2 == 0:
            if s > -1.0:
                s += -.2
            elif s > -120.0:
                s *= 1.06

        if classes is not None:
            classes_old = classes

        classes = torch.argmin(pairwise_bregman(X, centers, phi, shape), axis=1)
        if classes_old is not None and classes is not None and torch.equal(classes_old, classes):
            convergence_cnt += 1
        else:
            convergence_cnt = 0


        #plot objective func
        # if iter % 1000 == 0:
        #     obj_list = np.append(obj_list, VI(len(centers), y, len(centers), classes))
        #     iter_list = np.append(iter_list, iter)
        #     visualize_lineplot(obj_list, iter_list, line_names=['objective'], save_path_list=['/home/adi/Duke/Clustering_Research/visualizations_91/VI_bregman_' + str(s_0) + '_' + str(i) + '_' + str(convergence_threshold) + '_' + str(X.shape[1]) + '.png'])

        if convergence_cnt == convergence_threshold:
            #print("point assignments have converged")
            break

    #assign cluster labels
    #centers = torch.Tensor(np.sort(centers, axis=0)) #TODO: new line
    bregman_dist = pairwise_bregman(X, centers, phi, shape) #n x k distance matrix
    classes = torch.argmin(bregman_dist, axis=1)

    return failed_to_converge, classes, centers, s, iter, time.time() - start


'''
X: n x m
centers: k x m
weights: n x k
phi --> function
'''
def bregman_update(X, centers, weights, phi, n_epochs, lr, iterative):
    if iterative:
        k, m = centers.size()

        for i in range(n_epochs):
            for j in range(k):
                second_deriv = phi[2](centers[j:j+1,:])
                gradient = torch.sum(torch.mul(torch.t(torch.matmul(centers[j,:] - X, second_deriv)), weights[:,j]), axis=1)
                centers[j,:] -= lr*gradient

    else:
        centers = torch.matmul(torch.t(weights), X) #(k x n) x (n x m) --> k x m
        weight_sum = torch.sum(weights, axis=0) #k-dim vector
        centers = torch.div(torch.t(centers), weight_sum) #m x k, k -->  m x k
        centers =  torch.t(centers) #k x m

    return centers
