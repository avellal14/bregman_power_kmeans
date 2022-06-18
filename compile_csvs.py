import os
import csv
import pandas as pd
import numpy as np

def compile_csvs(csv_dir):
    csv_list = os.listdir(csv_dir)

    line_list = []
    for csv_nm in sorted(csv_list):
        test_dict = eval(csv_nm[:-4])
        s_0 = test_dict['s_0']
        d = test_dict['data_params']['n_features']

        csv_df = pd.read_csv(os.path.join(csv_dir, csv_nm))
        mean_VI_og = csv_df['Mean VI K-Means'][0]
        se_VI_og = csv_df['SE VI K-Means'][0]

        mean_VI_power = csv_df['Mean VI Power K-Means'][0]
        se_VI_power = csv_df['SE VI Power K-Means'][0]

        mean_VI_bregman = csv_df['Mean VI Bregman Power K-Means Iterative'][0]
        se_VI_bregman = csv_df['SE VI Bregman Power K-Means Iterative'][0]
        
        line_list += [[s_0, d, round(mean_VI_og,3), round(mean_VI_power, 3), round(mean_VI_bregman,3)]]

    summary_csv = os.path.join(csv_dir, 'summary.csv') 
    with open(summary_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for line in line_list:
            writer.writerow(line)

if __name__ == "__main__":
    compile_csvs(csv_dir='experiment_dir'))
