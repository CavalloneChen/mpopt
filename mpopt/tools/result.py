#!/usr/bin/env python
"""Tools for results presentation and analysis

Functions:
    read_result:    unified API to read optimization results.
    stats_compare:  make statistical comparation of two results, print in terminal.
    average_rank:   rank and compare multiple results, print in terminal.
"""

import os
import numpy as np
import scipy.stats


import os 
import numpy as np
import pickle as pkl

import scipy.stats
from prettytable import PrettyTable

# data
# benchmark optimal
bias13 = list(range(-1400, 0, 100)) + list(range(100, 1500, 100))
bias17 = list(range(100, 3100, 100))

def load_result(*raw_paths):
    """
    Data loading of multiple results

    Inputs:
        *raw_paths (string): path or filename in \logs
            (filename is formatted like {}_{}_{}.pkl, last two fields defines the problem)

    Outputs:
        names   (list): data names
        res     (list of np.ndarray): optimization results
        cst     (list of np.ndarray): cost times
        ...     (adding)
    """
    paths = []
    for path in raw_paths:
        if not os.path.isabs(path):
            paths.append(os.path.join("/home/lyf/projects/fireworks_algorithms/logs/", path))
        else:
            paths.append(path)
    
    prob = raw_paths[0].split('/')[-1]
    prob = '_'.join(prob.split('_')[-2:])
    prob = prob[:-4]
    names = ['_'.join((filename.split('/')[-1]).split('_')[:-2]) for filename in raw_paths]
    
    res = []
    cst = []
    for path in paths:
        with open(path, 'rb') as f:
            res_dict = pkl.load(f)
        res.append(res_dict['res'])
        cst.append(res_dict['cst'])
     
    return names, prob, res, cst

def stats_compare(benchmark_name, *paths, **kwargs):
    """
    Statistically comparing of results 

    Inputs:
        *paths (string): path or filename in \logs
    """
    # handle selective kwargs
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.05

    # read data
    names, prob, res, cst = load_result(*paths)
    alg_num = len(names)

    means = [_.mean(axis=1).tolist() for _ in res]
    stds = [_.std(axis=1).tolist() for _ in res]
    #times = [_.mean(axis=1).tolist() for _ in cst]
    
    # remove bias
    if benchmark_name == 'CEC13':
        for idx in range(alg_num):
            means[idx] = [_[0] - _[1] for _ in zip(means[idx], bias13)]
    elif benchmark_name == 'CEC17':
        for idx in range(alg_num):
            means[idx] = [_[0] - _[1] for _ in zip(means[idx], bias17)]

    # benchmark num
    benchmark_num = len(means[0])

    if alg_num == 2:
        # If there're 2 algs, print results and statistical compare. 

        # stats analysis
        p_values = []
        signs = []
        for idx in range(benchmark_num):
            p =  scipy.stats.ranksums(res[0][idx,:], res[1][idx,:])[1]
            p_values.append(p)
            if p >= alpha:
                signs.append('=')
            else:
                if means[0][idx] < means[1][idx]:
                    signs.append('+')
                else:
                    signs.append('-')

        # prepare table
        tb = PrettyTable()
        tb.field_names = ['idx','alg1.mean','alg1.std','alg2.mean','alg2.std','P-value','Sig']
        win = 0
        lose = 0
        for idx in range(benchmark_num):
            row = [str(idx+1), 
                   '{:.3e}'.format(means[0][idx]),
                   '{:.3e}'.format(stds[0][idx]),
                   '{:.3e}'.format(means[1][idx]),
                   '{:.3e}'.format(stds[1][idx]),
                   '{:.2f}'.format(p_values[idx]),
                   signs[idx],]
            if row[-1] == '+':
                lose += 1
            elif row[-1] == '-':
                win += 1
            for idx in range(7):
                if row[-1] == '+':
                    row[idx] = '\033[1;31m' + row[idx] + '\033[0m'
                elif row[-1] == '-':
                    row[idx] = '\033[1;32m' + row[idx] + '\033[0m'
            tb.add_row(row)
        print("Comparing on {}: alg1: {}, alg2: {}".format(prob, names[0], names[1]))
        print("Win: {}, Lose: {}".format(win, lose))
        print(tb)
    else:
        # If there're more than 2 algs, print results and compute avgrank.
        
        # prepare table
        ranks = np.zeros((alg_num))
        tb = PrettyTable()
        fields = ['idx']
        for name in names:
            fields += [name+'.mean', name+'.std']
        tb.field_names = fields
        for benchmark_idx in range(benchmark_num):
            row = [str(benchmark_idx + 1)]
            for alg_idx in range(alg_num):
                row += ['{:.3e}'.format(means[alg_idx][benchmark_idx]),
                        '{:.3e}'.format(stds[alg_idx][benchmark_idx]),]
            
            sort_idx = np.argsort([means[_][benchmark_idx] for _ in range(alg_num)])
            alg_ranks = np.empty_like(sort_idx)
            alg_ranks[sort_idx] = np.arange(alg_num)
            
            min_j = sort_idx[0]

            row[2*min_j+1] = '\033[1;31m' + row[2*min_j+1] + '\033[0m'
            row[2*min_j+2] = '\033[1;31m' + row[2*min_j+2] + '\033[0m'
            
            tb.add_row(row)
            ranks += alg_ranks
        
        ranks /= benchmark_num
        rank_row = ['AvgRank']
        for alg_idx in range(alg_num):
            rank_row += ['{:.2f}'.format(1 + ranks[alg_idx]), '']
        tb.add_row(rank_row)

        print("Comparing on {}:".format(prob))
        print(tb)

if __name__ == '__main__':
    
    results = [#'/home/lyf/projects/fireworks_algorithms/logs/LoTFWA_CEC17_30D.pkl',
               os.path.abspath('./logs/CMAES_CEC17_30D.pkl'),
               '/home/lyf/projects/benchmarks/cec2017/results/EBOwithCMAR/EBOwithCMAR_CEC17_30D.pkl',
               '/home/lyf/projects/benchmarks/cec2017/results/jSO/jSO_CEC17_30D.pkl',
               '/home/lyf/projects/benchmarks/cec2017/results/LSHADE_SPACMA/LSHADE_SPACMA_CEC17_30D.pkl',
               #os.path.abspath('./logs/CMAES2_CEC17_30D.pkl'),
               #os.path.abspath('./logs/CMAES2cp_CEC17_30D.pkl'),
               #os.path.abspath('./logs/CMAES2cp01_CEC17_30D.pkl'),
               #os.path.abspath('./logs/CMAES2fcw_CEC17_30D.pkl'),
               #os.path.abspath('./logs/CMAES2fc_CEC17_30D.pkl'),
               os.path.abspath('./logs/HCFWA_CEC17_30D.pkl')
               ]
    stats_compare('CEC17', *results, alpha=0.1)
