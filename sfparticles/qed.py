from numba import njit, prange
from numpy import log10, sqrt, random
from math import floor
from scipy.constants import alpha, pi, m_e, hbar, c

@njit
def lookup_table_sigmoid(chi_e):
    return

_chi_range = (-3, 2) # 0.001-100
_chi_N = 256
_chi_delta = (_chi_range[1] - _chi_range[0]) / (_chi_N - 1)
_chi_table = random.rand(_chi_N) + 0.001

@njit(parallel=True, cache=True)
def lcfa_photon_prob(optical_depth, inv_gamma, chi_e, dt, N):
    factor = sqrt(3) / 2 / pi * alpha * m_e * c**2/hbar
    for ip in prange(N):
        log_chi_e = log10(chi_e[ip])
        if log_chi_e < _chi_range[0]:
            continue
        if log_chi_e > _chi_range[1]:
            idx = _chi_N - 1
        if _chi_range[0] < log_chi_e < _chi_range[1]:
            idx = floor((log_chi_e - _chi_range[0]) / _chi_delta)

        log_chi_e_left = 10**(_chi_range[0] + idx*_chi_delta)
        # linear interp
        k = (_chi_table[idx+1] - _chi_table[idx]) / _chi_delta
        prob_rate = _chi_table[idx] + k * (log_chi_e-log_chi_e_left)

        dtau = dt * inv_gamma[ip]

        optical_depth[ip] -= prob_rate * dtau
