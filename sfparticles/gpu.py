from os import environ
_use_gpu = False
if "SFPARTICLES_USE_GPU" in environ:
    if environ["SFPARTICLES_USE_GPU"] == "1":
        try:
            from numba import cuda
            import cupy as cp
        except ImportError as e:
            raise e
        
        print("using GPU")
        _use_gpu = True

if _use_gpu:
    from math import sqrt
    from .particles import c
    tpb = 128

    @cuda.jit
    def push_position_kernal( x, y, z, ux, uy, uz, inv_gamma, N, to_be_pruned, dt ):
        cdt = c*dt
        ip = cuda.grid(1)
        if ip < N:
            if to_be_pruned[ip]:
                return
            x[ip] += cdt * inv_gamma[ip] * ux[ip]
            y[ip] += cdt * inv_gamma[ip] * uy[ip]
            z[ip] += cdt * inv_gamma[ip] * uz[ip]
    def push_position( x, y, z, ux, uy, uz, inv_gamma, N, to_be_pruned, dt ):
        bpg = int(N/tpb) + 1
        push_position_kernal[bpg, tpb](x, y, z, ux, uy, uz, inv_gamma, N, to_be_pruned, dt)


    from .inline import boris_inline
    boris_gpu = cuda.jit(boris_inline)
    @cuda.jit
    def boris_kernal( ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, q, N, to_be_pruned, dt ) :
        ip = cuda.grid(1)
        if ip < N:
            if to_be_pruned[ip]:
                return
            ux[ip], uy[ip], uz[ip], inv_gamma[ip] = boris_gpu(
                ux[ip], uy[ip], uz[ip], Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], q, dt)

    def boris( ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, q, N, to_be_pruned, dt ):
        bpg = int(N/tpb) + 1
        boris_kernal[bpg, tpb](ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, q, N, to_be_pruned, dt)

    from .inline import LL_push_inline
    LL_push_gpu = cuda.jit(LL_push_inline)
    @cuda.jit
    def LL_push_kernal( ux, uy, uz, inv_gamma, chi_e,  N, to_be_pruned, dt ) :
        ip = cuda.grid(1)
        if ip < N:
            if to_be_pruned[ip]:
                return
            ux[ip], uy[ip], uz[ip], inv_gamma[ip] = LL_push_gpu(ux[ip], uy[ip], uz[ip], inv_gamma[ip], chi_e[ip], dt)
            
    def LL_push( ux, uy, uz, inv_gamma, chi_e,  N, to_be_pruned, dt ):
        bpg = int(N/tpb) + 1
        LL_push_kernal[bpg, tpb](ux, uy, uz, inv_gamma, chi_e,  N, to_be_pruned, dt)


    from .inline import calculate_chi_inline
    calculate_chi_gpu = cuda.jit(calculate_chi_inline)
    @cuda.jit
    def update_chi_kernal(Ex, Ey, Ez, Bx, By, Bz, ux, uy, uz, inv_gamma, chi_e, N, to_be_pruned):
        ip = cuda.grid(1)
        if ip < N:
            if to_be_pruned[ip]:
                return
            chi_e[ip] = calculate_chi_gpu(
                Ex[ip], Ey[ip], Ez[ip], 
                Bx[ip], By[ip], Bz[ip], 
                ux[ip], uy[ip], uz[ip], 
                inv_gamma[ip]
            )
    def update_chi(Ex, Ey, Ez, Bx, By, Bz, ux, uy, uz, inv_gamma, chi_e, N, to_be_pruned):
        bpg = int(N/tpb) + 1
        update_chi_kernal[bpg, tpb](Ex, Ey, Ez, Bx, By, Bz, ux, uy, uz, inv_gamma, chi_e, N, to_be_pruned)



    @cuda.jit
    def pick_hard_photon_kernal(event, delta, inv_gamma, threshold, N):
        ip = cuda.grid(1)
        if ip < N:
            if event[ip]:
                if delta[ip] / inv_gamma[ip] < threshold:
                    event[ip] = False
    def pick_hard_photon(event, delta, inv_gamma, threshold, N):
        bpg = int(N/tpb) + 1
        pick_hard_photon_kernal[bpg, tpb](event, delta, inv_gamma, threshold, N)


    @cuda.jit
    def photon_recoil_kernal(ux, uy, uz, inv_gamma, event, photon_delta, N, to_be_pruned):
        ip = cuda.grid(1)
        if ip < N:
            if to_be_pruned[ip]:
                return
            if event[ip]:
                ux[ip] *= 1 - photon_delta[ip]
                uy[ip] *= 1 - photon_delta[ip]
                uz[ip] *= 1 - photon_delta[ip]
                inv_gamma[ip] = 1 / sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2)
    def photon_recoil(ux, uy, uz, inv_gamma, event, photon_delta, N, to_be_pruned):
        bpg = int(N/tpb) + 1
        photon_recoil_kernal[bpg, tpb](ux, uy, uz, inv_gamma, event, photon_delta, N, to_be_pruned)

    @cuda.jit
    def create_photon_kernal(
        x_src, y_src, z_src, ux_src, uy_src, uz_src,
        x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
        inv_gamma_dst, photon_to_be_pruned,
        event_index, photon_delta, N_buffered, N_photon,
    ):
        ip = cuda.grid(1)
        if ip < N_photon:
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
    def create_photon(
        x_src, y_src, z_src, ux_src, uy_src, uz_src,
        x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
        inv_gamma_dst, photon_to_be_pruned,
        event_index, photon_delta, N_buffered, N_photon,
    ):
        bpg = int(N_photon/tpb) + 1
        create_photon_kernal[bpg, tpb](
            x_src, y_src, z_src, ux_src, uy_src, uz_src,
            x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
            inv_gamma_dst, photon_to_be_pruned,
            event_index, photon_delta, N_buffered, N_photon,
        )

    @cuda.jit
    def create_pair_kernal(
        x_src, y_src, z_src, ux_src, uy_src, uz_src,
        photon_to_be_pruned,
        x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
        inv_gamma_dst, pair_to_be_pruned,
        event_index, pair_delta, N_buffered, N_pair,
        inverse_delta,
    ):
        ip = cuda.grid(1)
        if ip < N_pair:
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
    def create_pair(
        x_src, y_src, z_src, ux_src, uy_src, uz_src,
        photon_to_be_pruned,
        x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
        inv_gamma_dst, pair_to_be_pruned,
        event_index, pair_delta, N_buffered, N_pair,
        inverse_delta,
    ):
        bpg = int(N_pair/tpb) + 1
        create_pair_kernal[bpg, tpb](
            x_src, y_src, z_src, ux_src, uy_src, uz_src,
            photon_to_be_pruned,
            x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
            inv_gamma_dst, pair_to_be_pruned,
            event_index, pair_delta, N_buffered, N_pair,
            inverse_delta,
        )

    @cuda.jit
    def _find_event_count_block(event, event_count_block):
        i = cuda.grid(1)
        if i < event.size:
            if event[i]:
                blockid = cuda.blockIdx.x
                cuda.atomic.inc(event_count_block, blockid, event.size)

    @cuda.jit
    def _find_event_index(event, event_count_block, event_index_start, event_index):
        '''
        block内的thread寻找event，
        每个thread对应一个event_index。
        [blocks,tpb]与find_event_count_block相同，
        block内的event数不会超过thread数。
        '''
        pos = cuda.grid(1)
        if pos < event.size:
            blockid = cuda.blockIdx.x
            tid = cuda.threadIdx.x # block内的tid
            nthread = int(event_count_block[blockid])
            nevent = nthread
            if nevent == 0 or tid > 0:
                return
            
            tpb = cuda.blockDim.x
            
            start = blockid * tpb
            end = min(event.size, start + tpb)
            
            if blockid == 0:
                idx = 0 + tid
            else:
                idx = event_index_start[blockid-1] + tid

            for ip in range(start, end):
                if event[ip]:
                    event_index[idx] = ip
                    idx += 1

    def find_event_index(d_event, N_event):
        N = d_event.size
        tpb = 16
        blocks = int(N/tpb) + 1
        
        d_event_count_block = cp.zeros(blocks, cp.uint64)
        _find_event_count_block[blocks, tpb](d_event, d_event_count_block)
        
        d_event_index_start = d_event_count_block.cumsum()
        d_event_index = cp.zeros(int(d_event_index_start[-1]), dtype=cp.int64)
        
        _find_event_index[blocks, tpb](d_event, d_event_count_block, d_event_index_start, d_event_index)
        
        return d_event_index

    @cuda.reduce
    def bool_sum_kernal(a, b):
        return a + b

    def bool_sum(a):
        return bool_sum_kernal(a.astype(cp.int64))
