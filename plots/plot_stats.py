import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import pearsonr 


def plot_stats():
    results = {}
    
    def update_results_by_name_values(id, file):
        id = id[:-len('.json')]
        id = id[:id.find('-2022')] + id[id.find('_grid'):]
        id = id.replace('_', '-')
        id = id[:-1] + 'seed' + id[-1]
        vals = id.split('-')
        
        for v in vals:
            if v.startswith('ex'):
                results[file]['n_examples'] = int(v[len('ex'):])
            elif v.startswith('env'):
                results[file]['n_envs'] = int(v[len('env'):])
            elif v.startswith('descr') and len(v) == len('descr'):
                results[file]['trained_with_description'] = 1
            elif v.startswith('grid'):
                results[file]['grid_size'] = v[len('grid'):]
            elif v.startswith('descr') and len(v) > len('descr'):
                results[file]['evaluated_with_description'] = int(True if v[len('descr'):] == 'True' else False)
            elif v.startswith('shift'):
                results[file]['domain_shift'] = int(True if v[len('shift'):] == 'True' else False)
            elif v.startswith('seed'):
                results[file]['seed'] = int(v[len('seed'):])
                
    def comptue_correlations(data):
        data['pearson'] = 0.0
        data['pvalue'] = 0.0
        for index, v in data.iterrows():
            if len(v['episode_lengths']) > 1:
                c, p = pearsonr(v['episode_lengths'], v['optimal_episode_lengths'])
                data['pearson'][index] = c
                data['pvalue'][index] = p
        return data.drop(['episode_lengths', 'optimal_episode_lengths'], axis=1, inplace=False)
    
    def compute_success_rate(data):
        data['success_rate'] = 0.0
        data['death_ratio'] = 0.0
        for index, v in data.iterrows():
            data['success_rate'][index] = v['n_success'] / v['n_runs']
            failures = v['n_failed'] + v['n_died']
            data['death_ratio'][index] = v['n_died'] / failures if failures > 0 else 0.0
        return data.drop(['n_success', 'n_runs', 'n_non_success', 'n_died', 'n_failed'], axis=1, inplace=False)

    for file in glob("tmp/*.json"):
        with open(file, "r") as f:
            results[file] = json.load(f)
            
            # correct for invalid data
            if results[file]['n_runs'] == 0:
                results[file]['n_runs'] = 20
            
            # create default value
            results[file]['trained_with_description'] = 0
            
            # create values for each experiment
            if 'ada' in file:
                results[file]['model_id'] = 'ada'
                id = file[len('tmp/ada')+1:]
                update_results_by_name_values(id, file)
            elif 'babbage' in file:
                results[file]['model_id'] = 'babbage'
                id = file[len('tmp/babbage')+1:]
                update_results_by_name_values(id, file)
            
    data = pd.DataFrame(results).T
    data = comptue_correlations(data)
    data = compute_success_rate(data)
    data = data.drop(['seed', 'n_envs'], axis=1, inplace=False)
    data = data.astype({
        "invalid_action_cnt": float,
        "blank_actions": float,
        "trained_with_description": float,
        "n_examples": float,
        "evaluated_with_description": float,
        "domain_shift": float,
        "pearson": float,
        "pvalue": float
    })
    
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    plt.grid(True)
    
    for i, group in data.groupby(["grid_size", "model_id", "n_examples"]):

        print(i)
        print(group.mean(numeric_only=True))

if __name__ == "__main__":
    plot_stats()
