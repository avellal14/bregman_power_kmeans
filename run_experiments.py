import csv
import os
import time
from datetime import datetime
from uuid import uuid4

import numpy as np
import torch

from utils import *
from euclidean_k_means import kmeans, power_kmeans
from bregman_k_means import power_kmeans_bregman


def setup_experiment(init_params, exp_dir):
    exp_file = exp_dir + str(init_params) + '.csv'
    data_lambda = lambda i: generate_points(random_state=i, **init_params['data_params']) 
    
    return exp_file, data_lambda 


def setup_experiment_group(exp_dir, exp_id):
    #setup experiment directory
    if not exp_id:
        exp_id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()) + '/'
    exp_dir = exp_dir + exp_id
    
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    
    return exp_dir


def run_experiment(init_params, exp_dir):
    start_exp = time.time()
    exp_file, data_func = setup_experiment(init_params, exp_dir)
    
    if os.path.isfile(exp_file): #if experiment has already been completed, exit
        return

    k = init_params['data_params']['centers']
    phi = get_phi(dist_to_phi(init_params['bregman_dist']))
    phi_gaussian = get_phi(dist_to_phi('gaussian'))

    iters_og = []
    times_og = []
    VIs_og = []

    iters_power = []
    times_power = []
    VIs_power = []

    iters_bregman_iterative = []
    times_bregman_iterative = []
    VIs_bregman_iterative = []

    iters_bregman = []
    times_bregman = []
    VIs_bregman = []


    for i in range(init_params['n_trials']):
        X, classes_true, centers_true = data_func(i)
        X = torch.tensor(X) #weird cast, clean up if possible
        X = X.to(torch.float64)

        centers_init = initcenters(X, k, random_state=i) #want to keep consistent center initialization
   
        #lloyd's, power k-means, bregman iterative, bregman closed form
        classes_og, centers_og, iter_og, time_og = kmeans(X, k, centers_init)
        classes_power, centers_power, s_final_power, iter_power, time_power = power_kmeans(X, init_params['s_0'], k, centers_init, classes_true, i)
        #classes_bregman_iterative, centers_bregman_iterative, s_final_bregman_iterative, iter_bregman_iterative, time_bregman_iterative = power_kmeans_bregman(phi, X, init_params['s_0'], k, centers_init_bregman, n_epochs=1, lr=0.01, iterative=True)
        classes_bregman, centers_bregman, s_final_bregman, iter_bregman, time_bregman = power_kmeans_bregman(phi, X, init_params['s_0'], k, centers_init, n_epochs=1, lr=0.01, iterative=False)   

        iters_og += [iter_og]
        iters_power += [iter_power]
        #iters_bregman_iterative += [iter_bregman_iterative]
        iters_bregman += [iter_bregman]

        times_og += [time_og]
        times_power += [time_power]
        #times_bregman_iterative += [time_bregman_iterative]
        times_bregman += [time_bregman]

        VIs_og += [VI(k, classes_true, k, classes_og)]
        VIs_power += [VI(k, classes_true, k, classes_power)]
        #VIs_bregman_iterative += [VI(k, classes_true, k, classes_bregman_iterative)]
        VIs_bregman += [VI(k, classes_true, k, classes_bregman)]

    iters_og = np.array(iters_og)
    times_og = np.array(times_og)
    VIs_og = np.array(VIs_og)

    iters_power = np.array(iters_power)
    times_power = np.array(times_power)
    VIs_power = np.array(VIs_power)

    #iters_bregman_iterative = np.array(iters_bregman_iterative)
    #times_bregman_iterative = np.array(times_bregman_iterative)
    #VIs_bregman_iterative = np.array(VIs_bregman_iterative)

    iters_bregman = np.array(iters_bregman)
    times_bregman = np.array(times_bregman)
    VIs_bregman = np.array(VIs_bregman)

    #print("nice: ", VIs_bregman)
    sqrt_n = np.sqrt(init_params['n_trials'])
    
    with open(exp_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Mean Iters K-Means", "SE Iters K-Means", "Mean Time Elapsed K-Means", "SE Time Elapsed K-Means", "Mean VI K-Means", "SE VI K-Means", "Mean Iters Power K-Means", "SE Iters Power K-Means", "Mean Time Elapsed Power K-Means", "SE Time Elapsed Power K-Means", "Mean VI Power K-Means", "SE VI Power K-Means", "Mean Iters Bregman Power K-Means Iterative", "SE Iters Bregman Power K-Means Iterative", "Mean Time Elapsed Bregman Power K-Means Iterative", "SE Time Elapsed Bregman Power K-Means Iterative", "Mean VI Bregman Power K-Means Iterative", "SE VI Bregman Power K-Means Iterative", "Mean Iters Bregman Power K-Means", "SE Iters Bregman Power K-Means", "Mean Time Elapsed Bregman Power K-Means", "SE Time Elapsed Bregman Power K-Means", "Mean VI Bregman Power K-Means", "SE VI Bregman Power K-Means"])
        writer.writerow([np.mean(iters_og), np.std(iters_og)/sqrt_n, np.mean(times_og), np.std(times_og)/sqrt_n, np.mean(VIs_og), np.std(VIs_og)/sqrt_n, np.mean(iters_power), np.std(iters_power)/sqrt_n, np.mean(times_power), np.std(times_power)/sqrt_n, np.mean(VIs_power), np.std(VIs_power)/sqrt_n, 
                         #np.mean(iters_bregman_iterative), np.std(iters_bregman_iterative)/sqrt_n, np.mean(times_bregman_iterative), np.std(times_bregman_iterative)/sqrt_n, np.mean(VIs_bregman_iterative), np.std(VIs_bregman_iterative)/sqrt_n,
                         np.mean(iters_bregman), np.std(iters_bregman)/sqrt_n, np.mean(times_bregman), np.std(times_bregman)/sqrt_n, np.mean(VIs_bregman), np.std(VIs_bregman)/sqrt_n])

    print("Experiment Done. Dimension: ", init_params['data_params']['n_features'], "s_0: ", init_params['s_0'], ", Time Elapsed (sec): ", time.time() - start_exp)


#exp_dir='/home/adi/hdd2/clustering_research/experiments/'

def run_experiment_group(exp_dir='/home/adi/Duke/Clustering_Research/experiments/', exp_id=None):
    exp_dir = setup_experiment_group(exp_dir, exp_id)

    # init_params = {
    #     'n_trials': 50,
    #     'bregman_dist': 'multinomial',
    #     'data_params': {
    #         'n_samples': 2500,
    #         'n_features': 2, 
    #         'center_box': (25, 125),
    #         'centers': 25,
    #         'data_dist': 'multinomial',
    #         'desired_variance': 0.625, 
    #     },
    #     's_0': -1.0
    # }

    init_params = {
        'n_trials': 50,
        'bregman_dist': 'gaussian',
        'data_params': {
            'n_samples': 2500,
            'n_features': 2, 
            'center_box': (25, 125),
            'centers': 25,
            'data_dist': 'gaussian',
            'desired_variance': 1.0,
        },
        's_0': -1.0
    }

   
    dims = [2, 5, 10, 20, 50, 100] #200
    s_0s = [-1.0, -3.0, -9.0] #-18.0

    for d in dims:
        init_params['data_params']['n_features'] = d
        init_params['data_params']['desired_variance'] = 0.5*d

        for s_0 in s_0s:
            init_params['s_0'] = s_0
            run_experiment(init_params, exp_dir)


if __name__ == "__main__":
    run_experiment_group()