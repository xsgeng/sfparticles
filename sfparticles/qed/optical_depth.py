from numba import njit, prange, void, float64, boolean, int64
from numpy import asarray, full, full_like, log, log10, sqrt, random, zeros
from .optical_depth_tables import integ_photon_prob_rate_from_table, photon_delta_from_chi_delta_table

@njit(
    void(float64[:], float64[:], float64[:], float64, int64, boolean[:], boolean[:], float64[:]),
    parallel=True, cache=False
)
def update_optical_depth(optical_depth, inv_gamma, chi_e, dt, N, to_be_pruned, event, delta):
    for ip in prange(N):
        if to_be_pruned[ip] or chi_e[ip] == 0.0:
            event[ip] = False
            delta[ip] = 0.0
            continue
        integ_prob_rate = integ_photon_prob_rate_from_table(chi_e[ip])
        dtau = dt * inv_gamma[ip]

        # reset if not set
        if optical_depth[ip] == 0.0:
            optical_depth[ip] = -log(1 - random.rand())

        optical_depth[ip] -= integ_prob_rate * dtau

        if optical_depth[ip] < 0:
            optical_depth[ip] = -log(1 - random.rand())
            event[ip] = True
            delta[ip] = photon_delta_from_chi_delta_table(chi_e[ip])
        else:
            event[ip] = False
            delta[ip] = 0.0