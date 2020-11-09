#!/usr/bin/env python
"""Tools for results presentation and analysis

Functions:
    compare: unified method to handle several result files.
    read_result: unified API to read optimization result file.
    stats_compare: make statistical comparation of two results, print in terminal.
    average_rank: rank and compare multiple results, print in terminal.
"""

import os
import json
import argparse
import numpy as np
import scipy.stats
from prettytable import PrettyTable

from ..benchmarks.benchmark import Benchmark

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='Paths for compared algorithms')
    parser.add_argument('-b', '--benchmark', default=None, help='Name of tested benchmark')
    parser.add_argument('-p', '--precision', default='1e-8', help='Precision in comparation')
    parser.add_argument('-a', '--alpha', default=0.05, help='Significant level for statistical comparation')

    return parser.parse_args()


def show_results(*paths, benchmark=None, **kwargs):
    """ Show comparation for some results """
    
    num_result = len(paths)

    if num_result == 2:
        stas_compare(*paths, benchmark=benchmark, **kwargs)
    elif num_result > 2:
        average_rank(*paths, benchmark=None, **kwargs)
    else:
        raise Exception("Need no less than two results.")


def stats_compare(*paths, benchmark=None, **kwargs):
    """Statistical comparation of two results in 'paths'. 
    
    #TODO: Add support for all functions.
    """
    
    # handle kwargs
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.05

    # read data
    name1, fit1, time1 = load_result(paths[0])
    name2, fit2, time2 = load_result(paths[1])

    # statistic analysis
    p_values = []
    signs = []
    
    err1 = fit1 - benchmark.bias[np.newaxis, :]
    err2 = fit2 - benchmark.bias[np.newaxis, :]
    
    mean1 = err1.mean(axis=1)
    mean2 = err2.mean(axis=1)
    std1 = err1.std(axis=1)
    std2 = err2.std(axis=1)

    for idx in range(benchmark.num_func):
        p =  scipy.stats.ranksums(err1[idx,:], err2[idx,:])[1]
        p_values.append(p)
        if p >= alpha:
                signs.append('=')
        else:
            if err1[idx] < err2[idx]:
                signs.append('+')
            else:
                signs.append('-')

    # print table
    tb = PrettyTable()
    tb.field_names = ['idx','alg1.mean','alg1.std','alg2.mean','alg2.std','P-value','Sig']
    win = 0
    lose = 0
    for idx in range(benchmark.num_func):
        row = [str(idx+1), 
               '{:.3e}'.format(mean1[idx]),
               '{:.3e}'.format(std1[idx]),
               '{:.3e}'.format(mean2[idx]),
               '{:.3e}'.format(std2[idx]),
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
    print("Comparing on {}: alg1: {}, alg2: {}".format(benchmark.name, name1, name2))
    print("Win: {}, Lose: {} (for alg2).".format(win, lose))
    print(tb)


def average_rank(*paths, benchmark=None, **kwargs):
    """ Compute and print average rank for multiple results in 'paths'. 
    
    #TODO: Add support for any function.
    """
    # handle kwargs
    precision = kwargs['precision'] if 'precision' in kwargs else 1e-8

    # read data
    num_alg = len(paths)
    names = []
    times = []
    fits = []
    
    for path in paths:
        name, fit, time = load_result(path)
        names.append(name)
        times.append(time)
        fits.append(fit)
    
    # compute means and stds
    means = np.array([[np.mean(fits[i][j,:]) for j in range(benchmark.num_func)] for i in range(num_alg)])
    stds = np.array([[np.std(fits[i][j,:]) for j in range(benchmark.num_func)] for j in range(num_alg)])

    # compute ranks
    ranks = np.zeros(benchmark.num_func, num_alg)
    for idx in range(benchmark.num_func):
        means = [np.mean(_[idx,:]) for _ in fits]
        ranks[idx, :] = ranking(means, precision)
    
    # print table
    tb = PrettyTable()
    
    fields = ['idx']
    for name in names:
        fields += [name+'.mean', name+'.std']
    tb.field_names = fields

    for benchmark_idx in range(benchmark.num_func):
        row = [str(benchmark_idx + 1)]
        for alg_idx in range(num_alg):
            tmp = ['{:.3e}'.format(means[alg_idx, benchmark_idx]),
                   '{:.3e}'.format(stds[alg_idx, benchmark_idx])]
            
            if ranks[benchmark_idx, alg_idx] == 1:
                tmp[0] = '\033[1;31m' + tmp[0] + '\033[0m'
                tmp[1] = '\033[1;31m' + tmp[1] + '\033[0m'

            row += tmp
            
        tb.add_row(row)
        
    ave_rank = np.mean(ranks, axis=0)
    rank_row = ['AvgRank']
    for alg_idx in range(num_alg):
        rank_row += ['{:.2f}'.format(ave_rank[alg_idx]), '']
    tb.add_row(rank_row)

    print("Comparing on {}:".format(prob))
    print(tb)


def ranking(scores, precision):
    
    if type(scores) is list:
        scores = np.array(scores)
    
    sort_idx = np.argsort(scores)
    ranks = np.ones_like(scores)
    
    for idx in range(1, scores.shape[0]):

        cur_idx = sort_idx[idx]
        pre_idx = sort_idx[idx-1]
        
        if scores[cur_idx] - scores[pre_idx] < precision:
            ranks[cur_idx] = ranks[pre_idx]
        else:
            ranks[cur_idx] = ranks[pre_idx] + 1

    return ranks


def load_result(path):
    with open(path, 'r') as f:
        res = json.loads(path)

    name = res['alg_name']
    fits = res['fits']
    times = res['times']

    return name, fits, times


if __name__ == '__main__':
    
    args = parsing()

    paths = args['paths']
    if len(paths) == 1:
        # compare all json results in the directory
        


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
