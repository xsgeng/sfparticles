from numba import njit, prange, void, float64, boolean, int64
from numpy import asarray, full, full_like, log, log10, sqrt, random, zeros
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