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

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def setup_experiment(init_params, exp_dir):
    init_params_name_copy = init_params['data_params'].copy()
    init_params_name_copy['s_0'] = init_params['s_0']
    del init_params_name_copy['center_box']
    del init_params_name_copy['center_coordinates']

    exp_file = exp_dir + str(init_params_name_copy) + '.csv'
    data_lambda = lambda i: generate_points(random_state=i, **init_params['data_params'])

    return exp_file, data_lambda


def setup_experiment_group(exp_dir, exp_id):
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

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
    ARIs_og = []
    NMIs_og = []

    iters_power = []
    times_power = []
    VIs_power = []
    ARIs_power = []
    NMIs_power = []

    iters_bregman_iterative = []
    times_bregman_iterative = []
    VIs_bregman_iterative = []
    ARIs_power = []
    NMIs_power = []

    iters_bregman = []
    times_bregman = []
    VIs_bregman = []
    ARIs_bregman = []
    NMIs_bregman = []


    i = 0
    cnt = 0

    start_trial = time.time()
    while cnt < init_params['n_trials']:
        X, classes_true, centers_true = data_func(i)
        X = torch.tensor(X) #weird cast, clean up if possible
        X = X.to(torch.float64)

        centers_init = initcenters(X, k, random_state=i) #want to keep consistent center initialization

        #lloyd's, power k-means, bregman iterative, bregman closed form
        classes_og, centers_og, iter_og, time_og = kmeans(X, k, centers_init)
        classes_power, centers_power, s_final_power, iter_power, time_power = power_kmeans(X, init_params['s_0'], k, centers_init, classes_true)
        #classes_bregman_iterative, centers_bregman_iterative, s_final_bregman_iterative, iter_bregman_iterative, time_bregman_iterative = power_kmeans_bregman(phi, X, init_params['s_0'], k, centers_init_bregman, n_epochs=1, lr=0.01, iterative=True)

        gamma_shape = init_params['data_params']['shape'] if 'shape' in init_params['data_params'].keys() else None
        failed_to_converge, classes_bregman, centers_bregman, s_final_bregman, iter_bregman, time_bregman = power_kmeans_bregman(phi, X, init_params['s_0'], k, centers_init, n_epochs=1, lr=0.01, iterative=False, convergence_threshold=init_params['convergence_threshold'], y=classes_true, shape=gamma_shape)
        i += 1

        if failed_to_converge:
            continue

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
        VIs_bregman += [VI(k, classes_true, k, classes_bregman)]

        ARIs_og += [adjusted_rand_score(classes_true, classes_og)]
        ARIs_power += [adjusted_rand_score(classes_true, classes_power)]
        ARIs_bregman += [adjusted_rand_score(classes_true, classes_bregman)]

        NMIs_og += [normalized_mutual_info_score(classes_true, classes_og)]
        NMIs_power += [normalized_mutual_info_score(classes_true, classes_power)]
        NMIs_bregman += [normalized_mutual_info_score(classes_true, classes_bregman)]

        cnt += 1

        #print("finished trial ", cnt, ". time elapsed: ", time.time() - start_trial)
        #print(VIs_power[-1], VIs_bregman[-1])
        start_trial = time.time()
        # print("og centers: ", centers_og, VIs_og[-1])
        # print("power centers: ", centers_power, VIs_power[-1])
        # print("bregman centers: ", centers_bregman, VIs_bregman[-1])

    iters_og = np.array(iters_og)
    times_og = np.array(times_og)
    VIs_og = np.array(VIs_og)
    ARIs_og = np.array(ARIs_og)
    NMIs_og = np.array(NMIs_og)

    iters_power = np.array(iters_power)
    times_power = np.array(times_power)
    VIs_power = np.array(VIs_power)
    ARIs_power = np.array(ARIs_power)
    NMIs_power = np.array(NMIs_power)

    iters_bregman = np.array(iters_bregman)
    times_bregman = np.array(times_bregman)
    VIs_bregman = np.array(VIs_bregman)
    ARIs_bregman = np.array(ARIs_bregman)
    NMIs_bregman = np.array(NMIs_bregman)

    #print("nice: ", VIs_bregman)
    sqrt_n = np.sqrt(init_params['n_trials'])

    with open(exp_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # writer.writerow(["Mean Iters K-Means", "SE Iters K-Means", "Mean Time Elapsed K-Means", "SE Time Elapsed K-Means", "Mean VI K-Means", "SE VI K-Means", "Mean Iters Power K-Means", "SE Iters Power K-Means", "Mean Time Elapsed Power K-Means", "SE Time Elapsed Power K-Means", "Mean VI Power K-Means", "SE VI Power K-Means", "Mean Iters Bregman Power K-Means", "SE Iters Bregman Power K-Means", "Mean Time Elapsed Bregman Power K-Means", "SE Time Elapsed Bregman Power K-Means", "Mean VI Bregman Power K-Means", "SE VI Bregman Power K-Means"])
        # writer.writerow([np.mean(iters_og), np.std(iters_og)/sqrt_n, np.mean(times_og), np.std(times_og)/sqrt_n, np.mean(VIs_og), np.std(VIs_og)/sqrt_n, np.mean(iters_power), np.std(iters_power)/sqrt_n, np.mean(times_power), np.std(times_power)/sqrt_n, np.mean(VIs_power), np.std(VIs_power)/sqrt_n,
        #                  np.mean(iters_bregman), np.std(iters_bregman)/sqrt_n, np.mean(times_bregman), np.std(times_bregman)/sqrt_n, np.mean(VIs_bregman), np.std(VIs_bregman)/sqrt_n])
        writer.writerow(["Mean VI K-Means", "SE VI K-Means", "Mean ARI K-Means", "SE ARI K-Means", "Mean NMI K-Means", "SE NMI K-Means", "Mean VI Power K-Means", "SE VI Power K-Means", "Mean ARI Power K-Means", "SE ARI Power K-Means", "Mean NMI Power K-Means", "SE NMI Power K-Means", "Mean VI Bregman Power K-Means", "SE VI Bregman Power K-Means", "Mean ARI Bregman Power K-Means", "SE ARI Bregman Power K-Means", "Mean NMI Bregman Power K-Means", "SE NMI Bregman Power K-Means"])
        writer.writerow([np.mean(VIs_og), np.std(VIs_og)/sqrt_n, np.mean(ARIs_og), np.std(ARIs_og)/sqrt_n, np.mean(NMIs_og), np.std(NMIs_og)/sqrt_n, np.mean(VIs_power), np.std(VIs_power)/sqrt_n, np.mean(ARIs_power), np.std(ARIs_power)/sqrt_n, np.mean(NMIs_power), np.std(NMIs_power)/sqrt_n, np.mean(VIs_bregman), np.std(VIs_bregman)/sqrt_n, np.mean(ARIs_bregman), np.std(ARIs_bregman)/sqrt_n, np.mean(NMIs_bregman), np.std(NMIs_bregman)/sqrt_n])

    print("Experiment Done. Dimension: ", init_params['data_params']['n_features'], "s_0: ", init_params['s_0'], ", Time Elapsed (sec): ", time.time() - start_exp)


#exp_dir='/home/adi/hdd2/clustering_research/experiments/'

def run_experiment_group(init_params, s_0s, exp_dir='/home/adi/Duke/Clustering_Research/experiments/', exp_id=None):
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

    # init_params = {
    #     'n_trials': 50,
    #     'bregman_dist': 'gaussian',
    #     'data_params': {
    #         'n_samples': 2500,
    #         'n_features': 2,
    #         'center_box': (25, 125),
    #         'centers': 25,
    #         'data_dist': 'gaussian',
    #         'desired_variance': 1.0,
    #     },
    #     's_0': -1.0,
    #     'convergence_threshold': 5
    # }

    #dims = [2]#[2, 5, 10, 20, 50, 100] #200
    #convergence_thresholds = [5,10]#[1,2,5,10]
    # s_0s = [-0.2, -1.0, -3.0, -9.0] #-18.0


    for s_0 in s_0s:
        init_params['s_0'] = s_0
        print("running experiment, s_0: ", init_params['s_0'])
        run_experiment(init_params, exp_dir)

    # for d in dims:
    #     init_params['data_params']['n_features'] = d
    #     init_params['data_params']['desired_variance'] = 0.5*d
    #
    #     for s_0 in s_0s:
    #         init_params['s_0'] = s_0
    #
    #         for threshold in convergence_thresholds:
    #             init_params['convergence_threshold'] = threshold
    #             run_experiment(init_params, exp_dir)


if __name__ == "__main__":
    k = 3
    d = 2
    init_params_gaussian = {
        'n_trials': 250,
        'bregman_dist': 'gaussian',
        'data_params': {
            'n_samples': 99, #they did 100 in robust bregman clustering, but 99 is divisible by 3
            'n_features': d,
            'center_box': (1, 40),
            'center_coordinates': np.array([[10]*d, [20]*d, [40]*d]),
            'centers': k,
            'data_dist': 'gaussian',
            'desired_variance': 16.0,
        },
        'convergence_threshold': 5
    }


    init_params_binomial = {
            'n_trials': 250,
            'bregman_dist': 'multinomial',
            'data_params': {
                'n_samples': 99, #they did 100 in robust bregman clustering, but 99 is divisible by 3
                'n_features': d,
                'center_box': (1, 40),
                'center_coordinates': np.array([[10]*d, [20]*d, [40]*d]),
                'centers': k,
                'data_dist': 'multinomial',
                'desired_variance': None, #isn't used with multinomial
            },
            'convergence_threshold': 10
    }


    init_params_poisson = {
            'n_trials': 250,
            'bregman_dist': 'poisson',
            'data_params': {
                'n_samples': 99, #they did 100 in robust bregman clustering, but 99 is divisible by 3
                'n_features': d,
                'center_box': (1, 40),
                'center_coordinates': np.array([[10]*d, [20]*d, [40]*d]),
                'centers': k,
                'data_dist': 'poisson',
                'desired_variance': None, #isn't used with poisson
            },
            'convergence_threshold': 10
        }

    init_params_gamma = {
            'n_trials': 250,
            'bregman_dist': 'gamma',
            'data_params': {
                'n_samples': 99, #they did 100 in robust bregman clustering, but 99 is divisible by 3
                'n_features': 2,
                'center_box': (1, 40), #they did 10,20,40 in robust bregman clustering: https://arxiv.org/pdf/1812.04356.pdf
                'centers': 3,
                'center_coordinates': np.array([[10,10], [20,20], [40,40]]),
                'data_dist': 'gamma',
                'desired_variance': None, #isn't used with gamma
                'shape': 3.0,
            },
            'convergence_threshold': 10
    }


    #run_experiment_group(init_params, s_0s=[-3.0, -9.0], exp_dir='/home/adi/Duke/Clustering_Research/experiments_1227/')
    # run_experiment_group(init_params_gaussian, s_0s=[-0.2, -1.0, -3.0, -9.0], exp_dir='/home/adi/Duke/Clustering_Research/experiments_0103/')
    run_experiment_group(init_params_binomial, s_0s=[-0.2, -1.0, -3.0, -9.0], exp_dir='/home/adi/Duke/Clustering_Research/experiments_0103/')
    # run_experiment_group(init_params_poisson, s_0s=[-0.2, -1.0, -3.0, -9.0], exp_dir='/home/adi/Duke/Clustering_Research/experiments_0103/')
    # run_experiment_group(init_params_gamma, s_0s=[-0.2, -1.0, -3.0, -9.0], exp_dir='/home/adi/Duke/Clustering_Research/experiments_0103/')
