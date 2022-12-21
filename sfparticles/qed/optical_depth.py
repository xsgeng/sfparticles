print("Using optical depth method.")
from numba import njit, prange, void, float64, boolean, int64
from numpy import log, random 
from .optical_depth_tables import \
    integ_photon_prob_rate_from_table, integ_pair_prob_rate_from_table, \
    photon_delta_from_chi_delta_table, pair_delta_from_chi_delta_table, \
    _log_chi_range

_chi_min = 10.0**_log_chi_range[0]
@njit(
    void(float64[:], float64[:], float64[:], float64, int64, boolean[:], boolean[:], float64[:]),
    parallel=True, cache=False
)
def update_tau_e(tau_e, inv_gamma, chi_e, dt, N, to_be_pruned, event, delta):
    for ip in prange(N):
        if to_be_pruned[ip] or chi_e[ip]  < _chi_min:
            event[ip] = False
            delta[ip] = 0.0
            continue
        integ_prob_rate = integ_photon_prob_rate_from_table(chi_e[ip])
        dtau = dt * inv_gamma[ip]

        # reset if not set
        if tau_e[ip] == 0.0:
            tau_e[ip] = -log(1 - random.rand())

        tau_e[ip] -= integ_prob_rate * dtau

        if tau_e[ip] < 0:
            tau_e[ip] = -log(1 - random.rand())
            event[ip] = True
            delta[ip] = photon_delta_from_chi_delta_table(chi_e[ip])
        else:
            event[ip] = False
            delta[ip] = 0.0

@njit(
    void(float64[:], float64[:], float64[:], float64, int64, boolean[:], boolean[:], float64[:]),
    parallel=True, cache=False
)
def update_tau_gamma(tau_gamma, inv_gamma, chi_gamma, dt, N, to_be_pruned, event, delta):
    for ip in prange(N):
        if to_be_pruned[ip] or chi_gamma[ip] < _chi_min:
            event[ip] = False
            delta[ip] = 0.0
            continue
        integ_prob_rate = integ_pair_prob_rate_from_table(chi_gamma[ip])
        dtau = dt * inv_gamma[ip]

        # reset if not set
        if tau_gamma[ip] == 0.0:
            tau_gamma[ip] = -log(1 - random.rand())

        tau_gamma[ip] -= integ_prob_rate * dtau

        if tau_gamma[ip] < 0:
            tau_gamma[ip] = -log(1 - random.rand())
            event[ip] = True
            delta[ip] = pair_delta_from_chi_delta_table(chi_gamma[ip])
        else:
            event[ip] = False
            delta[ip] = 0.0