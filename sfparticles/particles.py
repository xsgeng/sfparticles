from enum import Enum, auto
from typing import Tuple, Union
import numpy as np
from scipy.constants import c, m_e, e, hbar, epsilon_0, pi
from numba import njit, prange, float64, int64, void, boolean, uint64
from .fields import Fields

from .qed import _use_optical_depth
if _use_optical_depth:
    from .qed.optical_depth import update_tau_e, update_tau_gamma
else:
    from .qed.rejection_sampling import photon_from_rejection_sampling, pair_from_rejection_sampling

class RadiationReactionType(Enum):
    """
    `LL` : approximated Landau-Lifshitz equation for gamma >> 1
    `cLL` : quantum-corrected Landau-Lifshitz equation
    """
    NONE = auto()
    PHOTON = auto()
    LL = auto()
    CLL = auto()
    

class Particles(object):
    def __init__(
        self, name : str,
        q: int, m: float, N: int = 0,
        props : Tuple = None,
        RR : RadiationReactionType = RadiationReactionType.PHOTON,
        has_spin = False,
        push = True,
    ) -> None:
        """
        ## 初始化粒子

        `q` : int
            电荷，正负电子分别为±1
        `m` : float
            质量，以电子质量为单位
        `N` : int
            粒子数
        `props` : Tuple(x, y, z, ux, uy, uz, [sx, sy, sz])
            粒子的初始状态，可包含自旋矢量。如果`has_spin=True`且未给出sx, sy, sz，默认sz=1
            这些属性向量的长度为buffer的长度，前N个为粒子的属性。
        `RR` : RadiationReactionType or None
            辐射反作用类型。对m=0光子无效。
                `None` : 无RR
                `RadiationReactionType.photon` : 光子产生辐射反作用
                `RadiationReactionType.LL` : LL方程
                `RadiationReactionType.cLL` : 量子修正的LL方程
        `has_spin` : bool
            是否包含自旋
        `push` : Bool
            是否模拟该粒子的运动, 默认True
        """
        self.name = name
        self.q = q * e
        self.m = m * m_e
        self.has_spin = has_spin
        self.RR = RR
        self.push = push
        N = int(N)

        self.attrs = []

        assert m >= 0, 'negative mass'
        if m == 0:
            assert q == 0, 'photons cannot have mass'

        assert isinstance(RR, RadiationReactionType), 'RR must be RadiationReactionType'

        if props is None:
            if m == 0:
                assert N == 0, "cannot initialize photons with only N without props."
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
        self.attrs += ['x', 'y', 'z', 'ux', 'uy', 'uz']

        if self.has_spin:
            self.sx = sx
            self.sy = sy
            self.sz = sz
            self.attrs += ['sx', 'sy', 'sz']

        # gamma factor
        if m > 0:
            self.inv_gamma = 1./np.sqrt( 1 + self.ux**2 + self.uy**2 + self.uz**2 )
        else:
            assert ( self.ux**2 + self.uy**2 + self.uz**2 > 0).all(), "photon momentum cannot be 0"
            self.inv_gamma = 1./np.sqrt( self.ux**2 + self.uy**2 + self.uz**2 )
        self.attrs += ['inv_gamma']
        
        # quantum parameter
        self.chi = np.zeros(N)
        self.attrs += ['chi']



        # fields at particle positions
        self.Ez = np.zeros(N)
        self.Ex = np.zeros(N)
        self.Ey = np.zeros(N)
        self.Bz = np.zeros(N)
        self.Bx = np.zeros(N)
        self.By = np.zeros(N)
        self.attrs += ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']

        # buffer
        self.N_buffered = N
        self.buffer_size = N

        # prune flag
        self._to_be_pruned = np.full(N, False)
        self.attrs += ['_to_be_pruned']

    @property
    def Npart(self):
        return bool_sum(~self._to_be_pruned)

    @property
    def gamma(self):
        return 1.0 / self.inv_gamma

    def set_photon(self, photon):
        assert self.m > 0, 'photon cannot radiate photon'
        assert self.RR != RadiationReactionType.LL and self.RR != RadiationReactionType.CLL, 'LL equation does not radiate photon'
        assert isinstance(photon, Particles), 'photon must be Particle class'
        assert photon.m == 0 and photon.q == 0, 'photon must be m=0 and q=0'
        self.radiating = True
        self.photon = photon.name
        self.event = np.full(self.buffer_size, False)
        self.event_index = np.zeros(self.buffer_size, dtype=int)
        self.photon_delta = np.zeros(self.buffer_size)
        self.attrs += ["event", "event_index", "photon_delta"]
        if _use_optical_depth:
            self.tau = np.zeros(self.buffer_size)
            self.attrs += ['tau']
        
        
    def set_pair(self, electron, positron):
        assert self.m == 0, 'massive particle cannot create BW pair'
        assert isinstance(electron, Particles) and isinstance(positron, Particles), 'pair must be tuple or list of Particle class'
        assert electron.m == m_e and electron.q == -e, f'first of the pair must be electron'
        assert positron.m == m_e and positron.q ==  e, f'second of the pair must be positron'
        self.bw = True
        self.bw_electron = electron.name
        self.bw_positron = positron.name
        self.event = np.full(self.buffer_size, False)
        self.event_index = np.zeros(self.buffer_size, dtype=int)
        self.pair_delta = np.zeros(self.buffer_size)
        self.attrs += ["event", "event_index", "pair_delta"]
        if _use_optical_depth:
            self.tau = np.zeros(self.buffer_size)
            self.attrs += ['tau']
        
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
            if self.RR == RadiationReactionType.LL:
                LL_push(
                    self.ux, self.uy, self.uz, 
                    self.inv_gamma, self.chi,  
                    self.N_buffered, self._to_be_pruned, dt
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
        update_chi(
            self.Ex, self.Ey, self.Ez, 
            self.Bx, self.By, self.Bz, 
            self.ux, self.uy, self.uz,
            self.inv_gamma, self.chi, self.N_buffered, self._to_be_pruned, 
        )


    def _photon_event(self, dt):
        if _use_optical_depth:
            update_tau_e(self.tau, self.inv_gamma, self.chi, dt, self.buffer_size, self._to_be_pruned, self.event, self.photon_delta)
        else:
            photon_from_rejection_sampling(self.inv_gamma, self.chi, dt, self.N_buffered, self._to_be_pruned, self.event, self.photon_delta )
        # RR
        if self.RR == RadiationReactionType.PHOTON:
            photon_recoil(self.ux, self.uy, self.uz, self.inv_gamma, self.event, self.photon_delta, self.N_buffered, self._to_be_pruned)
        return self.event, self.photon_delta

    
    def _pair_event(self, dt):
        if _use_optical_depth:
            update_tau_gamma(self.tau, self.inv_gamma, self.chi, dt, self.buffer_size, self._to_be_pruned, self.event, self.pair_delta)
        else:
            pair_from_rejection_sampling(self.inv_gamma, self.chi, dt, self.N_buffered, self._to_be_pruned, self.event, self.pair_delta )
        return self.event, self.pair_delta

    
    def _pick_hard_photon(self, threshold=2.0):
        pick_hard_photon(self.event, self.photon_delta, self.inv_gamma, threshold, self.N_buffered)
        
        
    def _create_photon(self, pho):
        if not self.event.any():
            return

        # events are already false when marked as pruned in QED
        N_photon = bool_sum(self.event)
        
        if hasattr(self, 'photon') and N_photon > 0:
            find_event_index(self.event, self.event_index, self.N_buffered)
            pho._extend(N_photon)
            create_photon(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                pho.x, pho.y, pho.z, pho.ux, pho.uy, pho.uz,
                pho.inv_gamma, pho._to_be_pruned,
                self.event_index, self.photon_delta, pho.N_buffered, N_photon,
            )
            
            pho.N_buffered += N_photon
        

    def _create_pair(self, ele, pos):
        if not self.event.any():
            return
        
        # events are already false when marked as pruned in QED
        N_pair = bool_sum(self.event)
        
        if hasattr(self, 'bw_electron'):
            find_event_index(self.event, self.event_index, self.N_buffered)
            ele._extend(N_pair)
            pos._extend(N_pair)
            
            create_pair(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                self._to_be_pruned,
                ele.x, ele.y, ele.z, ele.ux, ele.uy, ele.uz,
                ele.inv_gamma, ele._to_be_pruned,
                self.event_index, self.pair_delta, ele.N_buffered, N_pair,
                inverse_delta=False,
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
            buffer_size_new = self.buffer_size + int(self.N_buffered/4) + N_new

            for attr in self.attrs:
                # self.* = np.concatenate((self.*, append_buffer))
                getattr(self, attr).resize(buffer_size_new)

            self._to_be_pruned[-(buffer_size_new-self.buffer_size):] = True


            # new buffer size
            self.buffer_size = buffer_size_new
            

    def _prune(self):
        selected = ~self._to_be_pruned
        N = bool_sum(selected)
        for attr in self.attrs:
            setattr(self, attr, (getattr(self, attr))[selected])
        self.buffer_size = N
        self.N_buffered = N


@njit(void(*[float64[:]]*7, int64, boolean[:], float64), parallel=True, cache=False)
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


@njit(void(*[float64[:]]*10, float64, int64, boolean[:], float64), parallel=True, cache=False)
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


@njit(void(*[float64[:]]*5, int64, boolean[:], float64), parallel=True, cache=False)
def LL_push( ux, uy, uz, inv_gamma, chi_e,  N, to_be_pruned, dt ) :
    factor = -2/3 / (4*pi*epsilon_0) * e**2 * m_e * c / hbar**2 * dt
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        
        ux[ip] += factor * chi_e[ip]**2 * ux[ip]*inv_gamma[ip]
        uy[ip] += factor * chi_e[ip]**2 * uy[ip]*inv_gamma[ip]
        uz[ip] += factor * chi_e[ip]**2 * uz[ip]*inv_gamma[ip]
        inv_gamma[ip] = 1 / np.sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2)
    

@njit(void(*[float64[:]]*11, int64, boolean[:]), parallel=True, cache=False)
def update_chi(Ex, Ey, Ez, Bx, By, Bz, ux, uy, uz, inv_gamma, chi_e, N, to_be_pruned):
    factor = e*hbar / (m_e**2 * c**3)
    for ip in prange(N):
        if to_be_pruned[ip]:
            continue
        gamma = 1.0 / inv_gamma[ip]
        chi_e[ip] = factor * np.sqrt(
            (gamma*Ex[ip] + (uy[ip]*Bz[ip] - uz[ip]*By[ip])*c)**2 +
            (gamma*Ey[ip] + (uz[ip]*Bx[ip] - ux[ip]*Bz[ip])*c)**2 +
            (gamma*Ez[ip] + (ux[ip]*By[ip] - uy[ip]*Bx[ip])*c)**2 -
            (ux[ip]*Ex[ip] + uy[ip]*Ey[ip] + uz[ip]*Ez[ip])**2
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
            inv_gamma[ip] = 1 / np.sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2)
        

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
        
        inv_gamma_dst[idx_dst] = 1.0 / np.sqrt(ux_dst[idx_dst]**2 + uy_dst[idx_dst]**2 + uz_dst[idx_dst]**2)
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
        
        inv_gamma_dst[idx_dst] = 1.0 / np.sqrt(1.0 + ux_dst[idx_dst]**2 + uy_dst[idx_dst]**2 + uz_dst[idx_dst]**2)
        # mark created pair as existing
        pair_to_be_pruned[idx_dst] = False
        # mark created pair as deleted
        photon_to_be_pruned[idx_src] = True
        

@njit(void(boolean[:], int64[:], int64))
def find_event_index(event, index, N):
    idx = 0
    for i in range(N):
        if event[i]:
            index[idx] = i
            idx += 1
            
@njit(uint64(boolean[:]))
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