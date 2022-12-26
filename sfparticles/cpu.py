from numba import njit, float64, void, int64, uint32, boolean, prange
from numpy import sqrt, zeros
from .particles import c, m_e

@njit(void(*[float64[:]]*7, int64, boolean[:], float64), parallel=True, cache=True)
def push_position( x, y, z, ux, uy, uz, inv_gamma, N, to_be_pruned, dt ):
    """
    Advance the particles' positions over `dt` using the momenta `ux`, `uy`, `uz`,
    """
    # Timestep, multiplied by c
    cdt = c*dt

    # Particle push (in parallel if threading is installed)
    for ip in prange(N) :
        if to_be_pruned[ip]:
            continue
        x[ip] += cdt * inv_gamma[ip] * ux[ip]
        y[ip] += cdt * inv_gamma[ip] * uy[ip]
        z[ip] += cdt * inv_gamma[ip] * uz[ip]


from .inline import boris_inline
boris_cpu = njit(boris_inline)
@njit(void(*[float64[:]]*10, float64, int64, boolean[:], float64), parallel=True, cache=False)
def boris( ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, q, N, to_be_pruned, dt ) :
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue

        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = boris_cpu(
            ux[ip], uy[ip], uz[ip], Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], q, dt)

from .inline import LL_push_inline
LL_push_cpu = njit(LL_push_inline)
@njit(void(*[float64[:]]*5, int64, boolean[:], float64), parallel=True, cache=False)
def LL_push( ux, uy, uz, inv_gamma, chi_e,  N, to_be_pruned, dt ) :
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = LL_push_cpu(ux[ip], uy[ip], uz[ip], inv_gamma[ip], chi_e[ip], dt)
        
from .inline import calculate_chi_inline
calculate_chi_cpu = njit(calculate_chi_inline)
@njit(void(*[float64[:]]*11, int64, boolean[:]), parallel=True, cache=False)
def update_chi(Ex, Ey, Ez, Bx, By, Bz, ux, uy, uz, inv_gamma, chi_e, N, to_be_pruned):
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        chi_e[ip] = calculate_chi_cpu(
            Ex[ip], Ey[ip], Ez[ip], 
            Bx[ip], By[ip], Bz[ip], 
            ux[ip], uy[ip], uz[ip], 
            inv_gamma[ip]
        )
        


@njit(void(*[float64[:]]*4, boolean[:], float64[:], int64, boolean[:]), parallel=True, cache=False)
def photon_recoil(ux, uy, uz, inv_gamma, event, photon_delta, N, to_be_pruned):
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        if event[ip]:
            ux[ip] *= 1 - photon_delta[ip]
            uy[ip] *= 1 - photon_delta[ip]
            uz[ip] *= 1 - photon_delta[ip]
            inv_gamma[ip] = 1 / sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2)
        

@njit(
    void(
        *[float64[:]]*12, 
        float64[:], boolean[:], 
        int64[:], float64[:], int64, int64,
    ), 
    parallel=True, cache=False
)
def create_photon(
    x_src, y_src, z_src, ux_src, uy_src, uz_src,
    x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
    inv_gamma_dst, photon_to_be_pruned,
    event_index, photon_delta, N_buffered, N_photon,
):
    for ip in prange(N_photon):
        idx_src = event_index[ip]
        idx_dst = N_buffered+ip
        x_dst[idx_dst] = x_src[idx_src]
        y_dst[idx_dst] = y_src[idx_src]
        z_dst[idx_dst] = z_src[idx_src]
        
        ux_dst[idx_dst] = photon_delta[idx_src] * ux_src[idx_src]
        uy_dst[idx_dst] = photon_delta[idx_src] * uy_src[idx_src]
        uz_dst[idx_dst] = photon_delta[idx_src] * uz_src[idx_src]
        
        inv_gamma_dst[idx_dst] = 1.0 / sqrt(ux_dst[idx_dst]**2 + uy_dst[idx_dst]**2 + uz_dst[idx_dst]**2)
        # mark created photon as existing
        photon_to_be_pruned[idx_dst] = False
        

@njit(
    void(
        *[float64[:]]*6, 
        boolean[:], 
        *[float64[:]]*6, 
        float64[:], boolean[:], 
        int64[:], float64[:], int64, int64,
        boolean,
    ), 
    parallel=True, cache=False
)
def create_pair(
    x_src, y_src, z_src, ux_src, uy_src, uz_src,
    photon_to_be_pruned,
    x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
    inv_gamma_dst, pair_to_be_pruned,
    event_index, pair_delta, N_buffered, N_pair,
    inverse_delta,
):
    for ip in prange(N_pair):
        idx_src = event_index[ip]
        idx_dst = N_buffered+ip
        x_dst[idx_dst] = x_src[idx_src]
        y_dst[idx_dst] = y_src[idx_src]
        z_dst[idx_dst] = z_src[idx_src]
        
        delta = pair_delta[idx_src]
        if inverse_delta: 
            delta = 1.0 - delta
        
        ux_dst[idx_dst] = delta * ux_src[idx_src]
        uy_dst[idx_dst] = delta * uy_src[idx_src]
        uz_dst[idx_dst] = delta * uz_src[idx_src]
        
        # TODO spin
        
        inv_gamma_dst[idx_dst] = 1.0 / sqrt(1.0 + ux_dst[idx_dst]**2 + uy_dst[idx_dst]**2 + uz_dst[idx_dst]**2)
        # mark created pair as existing
        pair_to_be_pruned[idx_dst] = False
        # mark created pair as deleted
        photon_to_be_pruned[idx_src] = True
        
         
@njit(int64[:](boolean[:]))
def find_event_index(event):
    event_index = zeros(event.sum(), dtype='int64')
    idx = 0
    for i in range(event.size):
        if event[i]:
            event_index[idx] = i
            idx += 1
    return event_index

@njit(int64(boolean[:]))
def bool_sum(event):
    ntotal = 0
    for ip in prange(len(event)):
        if event[ip]:
            ntotal += 1

    return ntotal

@njit((void(boolean[:], float64[:], float64[:], float64, int64)), parallel=True)
def pick_hard_photon(event, delta, inv_gamma, threshold, N):
    for ip in prange(N):
        if event[ip]:
            if delta[ip] / inv_gamma[ip] < threshold:
                event[ip] = False