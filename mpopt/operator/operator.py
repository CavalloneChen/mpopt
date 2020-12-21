""" General Operators for Swarms """

import numpy as np

from mpopt.tools.distribution import MultiVariateNormalDistribution as MVND

lb = -float('inf')
ub = float('inf')

""" Mapping Rules """
# Random Re-map
def random_map(samples, lb=-float('inf'), ub=float('inf')):
    in_bound = (samples > lb) * (samples < ub)
    if not in_bound.all():
        rand_samples = np.random.uniform(lb, ub, samples.shape)
        samples = in_bound * samples + (1 - in_bound) * rand_samples
    return samples

# Mirror Re-map
def mirror_map(samples):
    pass

# Mod Re-map
def mod_map(samples):
    pass


""" Explosion Methods """
def box_explode(idv, amp, num_spk, remap=random_map):
    dim = idv.shape[-1]
    bias = np.random.uniform(-1, 1, [num_spk, dim])
    spks = idv[np.newaxis, :] + amp * bias
    spks = remap(spks)
    return spks

def gaussian_explode(mvnd, nspk, remap=random_map):
    return mvnd.sample(nspk, remap)
    

""" Mutation Methods """
# Guided Mutation
def guided_mutate(idv, spk_pop, spk_fit, gm_ratio, remap=random_map):
    num_spk = spk_pop.shape[0]
    
    top_num = int(num_spk * gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    btm_idx = sort_idx[-top_num:]

    top_mean = np.mean(spk_pop[top_idx,:], axis=0)
    btm_mean = np.mean(spk_pop[btm_idx,:], axis=0)
    delta = top_mean - btm_mean

    ms = remap(idv + delta).reshape(1, -1)
    
    return ms

""" Selection Methods """
# Elite Selection
def elite_select(pop, fit, topk=1):
    if topk == 1:
        min_idx = np.argmin(fit)
        return pop[min_idx, :], fit[min_idx]
    else:
        sort_idx = np.argsort(fit)
        top_idx = sort_idx[:topk]
        return pop[top_idx, :], fit[top_idx]
