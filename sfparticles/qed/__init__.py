from numba import njit, prange
from numpy import asarray, full, full_like, log, log10, sqrt, random, zeros
from math import floor
from scipy.constants import alpha, pi, m_e, hbar, c


'''
Optical depth
'''
from .tables import integ_prob_rate_from_table, delta_from_chi_delta_table




@njit(parallel=True, cache=True)
def update_optical_depth(optical_depth, inv_gamma, chi_e, dt, N):
    event = full(N, False)
    delta = zeros(N)
    for ip in prange(N):
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
from .tables import prob_rate_from_table


@njit(parallel=True, cache=True)
def photon_from_rejection_sampling(inv_gamma, chi_e, dt, N):
    event = full(N, False)
    delta = zeros(N)
    for ip in prange(N):
        r1 = random.rand()
        r2 = random.rand()
        dtau = dt * inv_gamma[ip]

        # modified event generator by Gonoskov 2015
        prob_rate = 3*r1**2 * prob_rate_from_table(chi_e[ip], r1**3) * dtau
        if r2 < prob_rate:
            delta[ip] = r1**3
            event[ip] = True

    return event, delta