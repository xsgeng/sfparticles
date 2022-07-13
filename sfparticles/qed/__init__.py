from numba import njit, prange
from numpy import asarray, full, full_like, log, log10, sqrt, random, zeros
from math import floor
from scipy.constants import alpha, pi, m_e, hbar, c


'''
Optical depth
'''
from .tables import integ_prob_rate_from_table, delta_from_chi_delta_table

@njit(parallel=True, cache=False)
def update_optical_depth(optical_depth, inv_gamma, chi_e, dt, N, to_be_pruned):
    event = full(N, False)
    delta = zeros(N)
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        integ_prob_rate = integ_prob_rate_from_table(chi_e[ip])
        dtau = dt * inv_gamma[ip]
        optical_depth[ip] -= integ_prob_rate * dtau

        if optical_depth[ip] < 0:
            optical_depth[ip] = -log(1 - random.rand())
            event[ip] = True
            delta[ip] = delta_from_chi_delta_table(chi_e[ip])

    return event, delta

'''
Rejection_sampling
'''
from .tables import photon_prob_rate_from_table, pair_prob_rate_from_table

@njit(parallel=True, cache=False)
def photon_from_rejection_sampling(inv_gamma, chi_e, dt, N, to_be_pruned, event, delta):
    for ip in prange(N):
        if to_be_pruned[ip] or chi_e[ip] == 0.0:
            event[ip] = False
            continue
        r1, r2 = random.rand(2)
        dtau = dt * inv_gamma[ip]

        # modified event generator by Gonoskov 2015
        prob_rate = 3*r1**2 * photon_prob_rate_from_table(chi_e[ip], r1**3) * dtau
        if r2 < prob_rate:
            delta[ip] = r1**3
            event[ip] = True
        else:
            delta[ip] = 0.0
            event[ip] = False
            

    return event, delta

@njit(parallel=True, cache=False)
def pair_from_rejection_sampling(inv_gamma, chi_gamma, dt, N, to_be_pruned, event, delta):
    for ip in prange(N):
        if to_be_pruned[ip] or chi_gamma[ip] == 0.0:
            event[ip] = False
            continue
        r1, r2 = random.rand(2)
        dtau = dt * inv_gamma[ip]

        prob_rate = pair_prob_rate_from_table(chi_gamma[ip], r1) * dtau
        if r2 < prob_rate:
            delta[ip] = r1
            event[ip] = True
        else:
            delta[ip] = 0.0
            event[ip] = False

    return event, delta