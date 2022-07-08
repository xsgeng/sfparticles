from numba import njit, prange
from numpy import asarray, full, full_like, log, log10, sqrt, random, zeros
from math import floor
from scipy.constants import alpha, pi, m_e, hbar, c

from .tables import _chi_delta, _chi_N, _chi_range, _prob_rate_table

@njit
def prob_rate_from_chi_e(chi_e):
    log_chi_e = log10(chi_e)
    if log_chi_e < _chi_range[0]:
        return 0.0
    if log_chi_e > _chi_range[1]:
        idx = _chi_N - 1
    if _chi_range[0] < log_chi_e < _chi_range[1]:
        idx = floor((log_chi_e - _chi_range[0]) / _chi_delta)

    log_chi_e_left = _chi_range[0] + idx*_chi_delta
    # linear interp
    k = (_prob_rate_table[idx+1] - _prob_rate_table[idx]) / _chi_delta
    prob_rate = _prob_rate_table[idx] + k * (log_chi_e-log_chi_e_left)

    return prob_rate


# TODO
@njit
def delta_from_chi_delta_table(chi_e):
    return 0

@njit(parallel=True, cache=True)
def lcfa_photon_prob(optical_depth, inv_gamma, chi_e, dt, N):
    event = full(N, False)
    delta = zeros(N)
    for ip in prange(N):
        prob_rate = prob_rate_from_chi_e(chi_e[ip])
        dtau = dt * inv_gamma[ip]
        optical_depth[ip] -= prob_rate * dtau

        if optical_depth[ip] < 0:
            optical_depth[ip] = -log(1 - random.rand())
            event[ip] = True
            delta[ip] = delta_from_chi_delta_table(chi_e[ip])

    return event#, delta
