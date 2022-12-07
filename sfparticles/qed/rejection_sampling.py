from numba import njit, prange, void, float64, boolean, int64
from numpy import random
from .tables import photon_prob_rate_from_table, pair_prob_rate_from_table

@njit(void(float64[:], float64[:], float64, int64, boolean[:], boolean[:], float64[:]), parallel=True, cache=False)
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
            

@njit(void(float64[:], float64[:], float64, int64, boolean[:], boolean[:], float64[:]), parallel=True, cache=False)
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