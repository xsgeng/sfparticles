from typing import Tuple, Union
import numpy as np
from scipy.constants import c, m_e, e, hbar
from numba import njit, prange, guvectorize
from .fields import Fields

from .qed import photon_from_rejection_sampling, pair_from_rejection_sampling, update_optical_depth

class Particles(object):
    """
    ## 粒子属性
    - `q`, `m` : 电荷和质量，单位是e和m_e
    - `x`, `y`, `z` : 空间坐标
    - `ux`, `uy`, `uz` : 无量纲动量 p/mc
    """
    def __init__(
        self, name : str,
        q: int, m: float, N: int = 0,
        has_spin = False,
        props : Tuple = None,
    ) -> None:
        """
        ## 初始化粒子

        `q` : int
            电荷，正负电子分别为±1
        `m` : float
            质量，以电子质量为单位
        `N` : int
            粒子数
        `has_spin` : bool
            是否包含自旋
        `props` : Tuple(x, y, z, ux, uy, uz, [sx, sy, sz])
            粒子的初始状态，可包含自旋矢量。如果`has_spin=True`且未给出sx, sy, sz，默认sz=1
            这些属性向量的长度为buffer的长度，前N个为粒子的属性。
        `photon`, `pair` : Particles
            辐射光子和产生电子对的对象。
            pair = (electron, positron)
        """
        self.name = name
        self.q = q * e
        self.m = m * m_e
        self.has_spin = has_spin
        N = int(N)

        assert m >= 0, 'negative mass'

        if props is None:
            x = np.zeros(N)
            y = np.zeros(N)
            z = np.zeros(N)
            ux = np.zeros(N)
            uy = np.zeros(N)
            uz = np.zeros(N)
            if has_spin:
                sx = np.zeros(N)
                sy = np.zeros(N)
                sz = np.ones(N)

        if props:
            assert len(props) == 6 or len(props) == 9, 'given properties must has length of 6 or 9'

            for prop in props:
                if isinstance(prop, np.ndarray):
                    assert len(prop.shape) == 1, 'given property is not vector'
                    assert prop.shape[0] == N, 'given N does not match given property length'
                if isinstance(prop, (int, float)):
                    assert N == 1, 'given N does not match given property length'
                if isinstance(prop, (list, tuple)):
                    assert len(prop) == N, 'given N does not match given property length'
                
            for i, prop in enumerate(props):
                if isinstance(prop, (int, float)):
                    props[i] = [prop]

            x, y, z, ux, uy, uz = props[:6]
            if len(props) == 9 and has_spin:
                sx, sy, sz = props[7:]

        # position, momentum and spin vectors
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        self.ux = np.asarray(ux, dtype=np.float64)
        self.uy = np.asarray(uy, dtype=np.float64)
        self.uz = np.asarray(uz, dtype=np.float64)

        if self.has_spin:
            self.sx = sx
            self.sy = sy
            self.sz = sz

        # gamma factor
        if m > 0:
            self.inv_gamma = 1./np.sqrt( 1 + self.ux**2 + self.uy**2 + self.uz**2 )
        else:
            self.inv_gamma = 1./np.sqrt( self.ux**2 + self.uy**2 + self.uz**2 )
        
        # quantum parameter
        self.chi = np.zeros(N)
        self.optical_depth = -np.log(1 - np.random.rand(N))

        # fields at particle positions
        self.Ez = np.zeros(N)
        self.Ex = np.zeros(N)
        self.Ey = np.zeros(N)
        self.Bz = np.zeros(N)
        self.Bx = np.zeros(N)
        self.By = np.zeros(N)

        # buffer
        self.N_buffered = N
        self.buffer_size = N

        # prune flag
        self._to_be_pruned = np.full(N, False)

    @property
    def Npart(self):
        return (~self._to_be_pruned).sum()

    @property
    def gamma(self):
        return 1.0 / self.inv_gamma

    def set_photon(self, photon):
        assert self.m > 0, 'photon cannot radiate photon'
        assert isinstance(photon, Particles), 'photon must be Particle class'
        assert photon.m == 0 and photon.q == 0, 'photon must be m=0 and q=0'
        self.photon = photon
        self.event = np.full(self.buffer_size, False)
        self.event_index = np.zeros(self.buffer_size, dtype=int)
        self.photon_delta = np.full(self.buffer_size, 0.0)
        
        
    def set_pair(self, pair):
        assert self.m == 0, 'massive particle cannot create BW pair'
        assert isinstance(pair, (tuple, list)), 'pair must be tuple or list'
        assert isinstance(pair[0], Particles) and isinstance(pair[0], Particles), 'pair must be tuple or list of Particle class'
        assert len(pair) == 2, 'length of pair must be 2'
        assert pair[0].m == m_e and pair[0].q == -e, 'first of the pair must be electron'
        assert pair[1].m == m_e and pair[1].q ==  e, 'second of the pair must be positron'
        self.pair = pair
        self.event = np.full(self.buffer_size, False)
        self.event_index = np.zeros(self.buffer_size, dtype=int)
        self.pair_delta = np.full(self.buffer_size, 0.0)

        
    def _push_momentum(self, dt):
        if self.m > 0:
            if self.has_spin:
                boris_tbmt(
                    self.ux, self.uy, self.uz,
                    self.sx, self.sy, self.sz,
                    self.inv_gamma,
                    self.Ex, self.Ey, self.Ez, 
                    self.Bx, self.By, self.Bz,
                    self.q, self.N_buffered, self._to_be_pruned, dt
                )
            else:
                boris(
                    self.ux, self.uy, self.uz,
                    self.inv_gamma,
                    self.Ex, self.Ey, self.Ez, 
                    self.Bx, self.By, self.Bz,
                    self.q, self.N_buffered, self._to_be_pruned, dt
                )


    def _push_position(self, dt):
        push_position(
            self.x, self.y, self.z, 
            self.ux, self.uy, self.uz, 
            self.inv_gamma, 
            self.N_buffered, self._to_be_pruned, dt
        )


    def _eval_field(self, fields : Fields, t):
        fields.field_func(
            self.x, self.y, self.z, t, self.N_buffered, self._to_be_pruned,
            self.Ex, self.Ey, self.Ez, 
            self.Bx, self.By, self.Bz
        )


    def _calculate_chi(self):
        update_chi_e(
            self.Ex, self.Ey, self.Ez, 
            self.Bx, self.By, self.Bz, 
            self.ux, self.uy, self.uz,
            self.inv_gamma, self.chi, self.N_buffered, self._to_be_pruned, 
        )



    def _photon_event(self, dt):
        # event, photon_delta = update_optical_depth(self.optical_depth, self.inv_gamma, self.chi, dt, self.buffer_size, self._to_be_pruned)
        photon_from_rejection_sampling(self.inv_gamma, self.chi, dt, self.buffer_size, self._to_be_pruned, self.event, self.photon_delta )
        # RR
        radiation_reaction(self.ux, self.uy, self.uz, self.inv_gamma, self.event, self.photon_delta, self.N_buffered, self._to_be_pruned)
        return self.event, self.photon_delta

    def _pair_event(self, dt):
        pair_from_rejection_sampling(self.inv_gamma, self.chi, dt, self.buffer_size, self._to_be_pruned, self.event, self.pair_delta )
        return self.event, self.pair_delta
    
    def _create_photon(self):
        if not self.event.any():
            return

        pick_hard_photon(self.event, self.photon_delta, self.inv_gamma, 2.0, self.N_buffered)
        
        # events are already false when marked as pruned in QED
        N_photon = self.event.sum()
        
        if hasattr(self, 'photon') and N_photon > 0:
            find_event_index(self.event, self.event_index, self.buffer_size)
            pho = self.photon
            pho._extend(N_photon)
            create_photon(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                pho.x, pho.y, pho.z, pho.ux, pho.uy, pho.uz,
                pho.inv_gamma, pho._to_be_pruned,
                self.event_index, self.photon_delta, pho.N_buffered, N_photon,
            )
            
            pho.N_buffered += N_photon
        

    def _create_pair(self):
        if not self.event.any():
            return
        
        # events are already false when marked as pruned in QED
        N_pair = self.event.sum()
        
        if hasattr(self, 'pair'):
            find_event_index(self.event, self.event_index, self.buffer_size)
            ele = self.pair[0]
            pos = self.pair[1]
            ele._extend(N_pair)
            pos._extend(N_pair)
            
            create_pair(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                self._to_be_pruned,
                ele.x, ele.y, ele.z, ele.ux, ele.uy, ele.uz,
                ele.inv_gamma, ele._to_be_pruned,
                self.event_index, self.pair_delta, ele.N_buffered, N_pair,
            )
            create_pair(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                self._to_be_pruned,
                pos.x, pos.y, pos.z, pos.ux, pos.uy, pos.uz,
                pos.inv_gamma, pos._to_be_pruned,
                self.event_index, self.pair_delta, pos.N_buffered, N_pair,
                inverse_delta=True,
            )
            
            ele.N_buffered += N_pair
            pos.N_buffered += N_pair
            


    def _extend(self, N_new):
        # extend buffer
        if (self.buffer_size - self.N_buffered) < N_new:
            bufer_size_new = self.N_buffered + N_new
            append_buffer = np.zeros(bufer_size_new)
            for attr in ('x', 'y', 'z', 'ux', 'uy', 'uz', 'inv_gamma', 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'chi', 'optical_depth'):
                # self.* = np.concatenate((self.*, append_buffer))
                setattr(self, attr, np.concatenate((getattr(self, attr), append_buffer)))

            if hasattr(self, 'event'):
                self.event = np.concatenate((self.event, np.full(bufer_size_new, False)))
                self.event_index = np.concatenate((self.event_index, np.zeros(bufer_size_new, dtype=int)))
            if hasattr(self, 'photon_delta'):
                self.photon_delta = np.concatenate((self.photon_delta, append_buffer))
            if hasattr(self, 'pair_delta'):
                self.pair_delta = np.concatenate((self.pair_delta, append_buffer))

            self._to_be_pruned = np.concatenate((self._to_be_pruned, np.full(bufer_size_new, True)))

            if self.has_spin:
                self.sx = np.concatenate((self.sx, append_buffer))
                self.sy = np.concatenate((self.sy, append_buffer))
                self.sz = np.concatenate((self.sz, append_buffer))

            # new buffer size
            self.buffer_size += bufer_size_new
            

    def _prune(self):
        selected = ~self._to_be_pruned
        N = selected.sum()
        for attr in ('x', 'y', 'z', 'ux', 'uy', 'uz', 'inv_gamma', 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'chi', 'optical_depth', '_to_be_pruned'):
            setattr(self, attr, (getattr(self, attr))[selected])
        self.buffer_size = N
        self.N_buffered = N


@njit(parallel=True, cache=False)
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


@njit(parallel=True, cache=False)
def boris( ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, q, N, to_be_pruned, dt ) :
    """
    Advance the particles' momenta, using numba
    """
    efactor = q*dt/(2*m_e*c)
    bfactor = q*dt/(2*m_e)

    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        # E field
        ux_minus = ux[ip] + efactor * Ex[ip]
        uy_minus = uy[ip] + efactor * Ey[ip]
        uz_minus = uz[ip] + efactor * Ez[ip]
        # B field
        inv_gamma_minus = 1 / np.sqrt(1 + ux_minus**2 + uy_minus**2 + uz_minus**2)
        Tx = bfactor * Bx[ip] * inv_gamma_minus
        Ty = bfactor * By[ip] * inv_gamma_minus
        Tz = bfactor * Bz[ip] * inv_gamma_minus
        
        ux_prime = ux_minus + uy_minus * Tz - uz_minus * Ty
        uy_prime = uy_minus + uz_minus * Tx - ux_minus * Tz
        uz_prime = uz_minus + ux_minus * Ty - uy_minus * Tx

        Tfactor = 2 / (1 + Tx**2 + Ty**2 + Tz**2)
        Sx = Tfactor * Tx
        Sy = Tfactor * Ty
        Sz = Tfactor * Tz

        ux_plus = ux_minus + uy_prime * Sz - uz_prime * Sy
        uy_plus = uy_minus + uz_prime * Sx - ux_prime * Sz
        uz_plus = uz_minus + ux_prime * Sy - uy_prime * Sx

        ux[ip] = ux_plus + efactor * Ex[ip]
        uy[ip] = uy_plus + efactor * Ey[ip]
        uz[ip] = uz_plus + efactor * Ez[ip]
        inv_gamma[ip] = 1 / np.sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2)
    

# TODO
@njit(parallel=True, cache=False)
def boris_tbmt( ux, uy, uz, sx, sy, sz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, q, N, to_be_pruned, dt ) :
    """
    Advance the particles' momenta, using numba
    """
    efactor = q*dt/(2*m_e*c)
    bfactor = q*dt/(2*m_e)

    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        # E field
        ux_minus = ux[ip] + efactor * Ex[ip]
        uy_minus = uy[ip] + efactor * Ey[ip]
        uz_minus = uz[ip] + efactor * Ez[ip]
        # B field
        inv_gamma_minus = 1 / np.sqrt(1 + ux_minus**2 + uy_minus**2 + uz_minus**2)
        Tx = bfactor * Bx[ip] * inv_gamma_minus
        Ty = bfactor * By[ip] * inv_gamma_minus
        Tz = bfactor * Bz[ip] * inv_gamma_minus
        
        ux_prime = ux_minus + uy_minus * Tz - uz_minus * Ty
        uy_prime = uy_minus + uz_minus * Tx - ux_minus * Tz
        uz_prime = uz_minus + ux_minus * Ty - uy_minus * Tx

        Tfactor = 2 / (1 + Tx**2 + Ty**2 + Tz**2)
        Sx = Tfactor * Tx
        Sy = Tfactor * Ty
        Sz = Tfactor * Tz

        ux_plus = ux_minus + uy_prime * Sz - uz_prime * Sy
        uy_plus = uy_minus + uz_prime * Sx - ux_prime * Sz
        uz_plus = uz_minus + ux_prime * Sy - uy_prime * Sx

        ux[ip] = ux_plus + efactor * Ex[ip]
        uy[ip] = uy_plus + efactor * Ey[ip]
        uz[ip] = uz_plus + efactor * Ez[ip]
        inv_gamma[ip] = 1 / np.sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2)

        # TBMT
        ...
    

def vay( ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, q, N, to_be_pruned, dt ):
    """
    Push at single macroparticle, using the Vay pusher
    """
    # Set a few constants
    econst = q*dt/(m_e*c)
    bconst = 0.5*q*dt/m_e

    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        # Get the magnetic rotation vector
        taux = bconst*Bx[ip]
        tauy = bconst*By[ip]
        tauz = bconst*Bz[ip]
        tau2 = taux**2 + tauy**2 + tauz**2

        # Get the momenta at the half timestep
        uxp = ux[ip] + econst*Ex[ip] \
        + inv_gamma[ip] *( uy[ip]*tauz - uz[ip]*tauy )
        uyp = uy[ip] + econst*Ey[ip] \
        + inv_gamma[ip] *( uz[ip]*taux - ux[ip]*tauz )
        uzp = uz[ip] + econst*Ez[ip] \
        + inv_gamma[ip] *( ux[ip]*tauy - uy[ip]*taux )
        sigma = 1 + uxp**2 + uyp**2 + uzp**2 - tau2
        utau = uxp*taux + uyp*tauy + uzp*tauz

        # Get the new 1./gamma
        inv_gamma_f = np.sqrt(
            2./( sigma + np.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) ) )

        # Reuse the tau and utau arrays to save memory
        tx = inv_gamma_f*taux
        ty = inv_gamma_f*tauy
        tz = inv_gamma_f*tauz
        ut = inv_gamma_f*utau
        s = 1./( 1 + tau2*inv_gamma_f**2 )

        # Get the new u
        ux[ip] = s*( uxp + tx*ut + uyp*tz - uzp*ty )
        uy[ip] = s*( uyp + ty*ut + uzp*tx - uxp*tz )
        uz[ip] = s*( uzp + tz*ut + uxp*ty - uyp*tx )
        inv_gamma[ip] = 1 / np.sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2)


@njit(parallel=True, cache=False)
def update_chi_e(Ex, Ey, Ez, Bx, By, Bz, ux, uy, uz, inv_gamma, chi_e, N, to_be_pruned):
    gamma = 1 / inv_gamma
    factor = e*hbar / (m_e**2 * c**3)
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        chi_e[ip] = factor * np.sqrt(
            (gamma[ip]*Ex[ip] + (uy[ip]*Bz[ip] - uz[ip]*By[ip])*c)**2 +
            (gamma[ip]*Ey[ip] + (uz[ip]*Bx[ip] - ux[ip]*Bz[ip])*c)**2 +
            (gamma[ip]*Ez[ip] + (ux[ip]*By[ip] - uy[ip]*Bx[ip])*c)**2 -
            (ux[ip]*Ex[ip] + uy[ip]*Ey[ip] + uz[ip]*Ez[ip])**2
        )

@njit(parallel=True, cache=False)
def radiation_reaction(ux, uy, uz, inv_gamma, event, photon_delta, N, to_be_pruned):
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        if event[ip]:
            ux[ip] *= 1 - photon_delta[ip]
            uy[ip] *= 1 - photon_delta[ip]
            uz[ip] *= 1 - photon_delta[ip]
            inv_gamma[ip] = 1 / np.sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2)
        

@njit(parallel=True, cache=False)
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
        
        inv_gamma_dst[idx_dst] = 1.0 / np.sqrt(ux_dst[idx_dst]**2 + uy_dst[idx_dst]**2 + uz_dst[idx_dst]**2)
        # mark created photon as existing
        photon_to_be_pruned[idx_dst] = False
        
        
@njit(parallel=True, cache=False)
def create_pair(
    x_src, y_src, z_src, ux_src, uy_src, uz_src,
    photon_to_be_pruned,
    x_dst, y_dst, z_dst, ux_dst, uy_dst, uz_dst,
    inv_gamma_dst, pair_to_be_pruned,
    event_index, pair_delta, N_buffered, N_pair,
    inverse_delta = False,
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
        
        inv_gamma_dst[idx_dst] = 1.0 / np.sqrt(1.0 + ux_dst[idx_dst]**2 + uy_dst[idx_dst]**2 + uz_dst[idx_dst]**2)
        # mark created pair as existing
        pair_to_be_pruned[idx_dst] = False
        # mark created pair as deleted
        photon_to_be_pruned[idx_src] = True
        

@njit
def find_event_index(event, index, N):
    idx = 0
    for i in range(N):
        if event[i]:
            index[idx] = i
            idx += 1
            

@njit(parallel=True)
def pick_hard_photon(event, delta, inv_gamma, threshold, N):
    for ip in prange(N):
        if event[ip]:
            if delta[ip] / inv_gamma[ip] < threshold:
                event[ip] = False