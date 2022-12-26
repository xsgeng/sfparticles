from numba import njit, prange, void, float64, boolean, int64
from numpy import random
from .rejection_sampling_tables import photon_prob_rate_from_table, pair_prob_rate_from_table, _int_Ai_table, _Aip_table
from ..import _use_gpu
if _use_gpu:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
    from ..gpu import tpb
    gen = create_xoroshiro128p_states(tpb*5120, 0)
    photon_prob_rate_from_table = cuda.jit(photon_prob_rate_from_table)
    pair_prob_rate_from_table= cuda.jit(pair_prob_rate_from_table)
    _d_int_Ai_table = cuda.to_device(_int_Ai_table)
    _d_Aip_table= cuda.to_device(_Aip_table)

    @cuda.jit
    def photon_from_rejection_sampling_kernal(gen, int_Ai_table, Aip_table, inv_gamma, chi_e, dt, N, to_be_pruned, event, delta):
        ip = cuda.grid(1)
        if ip < N:
            if to_be_pruned[ip] or chi_e[ip] == 0.0:
                event[ip] = False
                return
            r1 = xoroshiro128p_uniform_float64(gen, ip%(tpb*5120))
            r2 = xoroshiro128p_uniform_float64(gen, ip%(tpb*5120))
            dtau = dt * inv_gamma[ip]

            # modified event generator by Gonoskov 2015
            prob_rate = 3*r1**2 * photon_prob_rate_from_table(chi_e[ip], r1**3, int_Ai_table, Aip_table) * dtau
            # prob_rate = 3*r1**2 * xoroshiro128p_uniform_float64(gen, ip%(tpb*5120)) * dtau
            if r2 < prob_rate:
                delta[ip] = r1**3
                event[ip] = True
            else:
                delta[ip] = 0.0
                event[ip] = False
                
    def photon_from_rejection_sampling(inv_gamma, chi_e, dt, N, to_be_pruned, event, delta):
        bpg = int(N/tpb) + 1
        photon_from_rejection_sampling_kernal[bpg,tpb](gen, _d_int_Ai_table, _d_Aip_table, inv_gamma, chi_e, dt, N, to_be_pruned, event, delta)

    @cuda.jit
    def pair_from_rejection_sampling_kernal(gen, int_Ai_table, Aip_table, inv_gamma, chi_gamma, dt, N, to_be_pruned, event, delta):
        ip = cuda.grid(1)
        if ip < N:
            if to_be_pruned[ip] or chi_gamma[ip] == 0.0:
                event[ip] = False
                return
            r1 = xoroshiro128p_uniform_float64(gen, ip%(tpb*5120))
            r2 = xoroshiro128p_uniform_float64(gen, ip%(tpb*5120))
            dtau = dt * inv_gamma[ip]

            prob_rate = pair_prob_rate_from_table(chi_gamma[ip], r1, int_Ai_table, Aip_table) * dtau
            # prob_rate = 3*r1**2 * xoroshiro128p_uniform_float64(gen, ip%(tpb*5120)) * dtau
            if r2 < prob_rate:
                delta[ip] = r1
                event[ip] = True
            else:
                delta[ip] = 0.0
                event[ip] = False

    def pair_from_rejection_sampling(inv_gamma, chi_e, dt, N, to_be_pruned, event, delta):
        bpg = int(N/tpb) + 1
        pair_from_rejection_sampling_kernal[bpg,tpb](gen, _d_int_Ai_table, _d_Aip_table, inv_gamma, chi_e, dt, N, to_be_pruned, event, delta)
else:
    photon_prob_rate_from_table = njit(photon_prob_rate_from_table)
    pair_prob_rate_from_table= njit(pair_prob_rate_from_table)

    @njit(void(float64[:], float64[:], float64, int64, boolean[:], boolean[:], float64[:]), parallel=True, cache=False)
    def photon_from_rejection_sampling(inv_gamma, chi_e, dt, N, to_be_pruned, event, delta):
        for ip in prange(N):
            if to_be_pruned[ip] or chi_e[ip] == 0.0:
                event[ip] = False
                continue
            r1, r2 = random.rand(2)
            dtau = dt * inv_gamma[ip]

            # modified event generator by Gonoskov 2015
            prob_rate = 3*r1**2 * photon_prob_rate_from_table(chi_e[ip], r1**3, _int_Ai_table, _Aip_table) * dtau
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

            prob_rate = pair_prob_rate_from_table(chi_gamma[ip], r1, _int_Ai_table, _Aip_table) * dtau
            if r2 < prob_rate:
                delta[ip] = r1
                event[ip] = True
            else:
                delta[ip] = 0.0
                event[ip] = False