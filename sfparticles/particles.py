from enum import Enum, auto
from typing import Tuple, Union
import numpy as np
from scipy.constants import c, m_e, e, hbar, epsilon_0, pi
from .fields import Fields

from .qed import _use_optical_depth
if _use_optical_depth:
    from .qed.optical_depth import update_tau_e, update_tau_gamma
else:
    from .qed.rejection_sampling import photon_from_rejection_sampling, pair_from_rejection_sampling

from .gpu import _use_gpu
if _use_gpu:
    import cupy as cp
    from cupy import resize
    from .gpu import push_position, boris, LL_push, update_chi, \
        pick_hard_photon, photon_recoil, create_pair, create_photon, \
        find_event_index, bool_sum
else:
    from numpy import resize
    from .cpu import push_position, boris, LL_push, update_chi, \
        pick_hard_photon, photon_recoil, create_pair, create_photon, \
        find_event_index, bool_sum

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
        self.bw = False
        self.radiating = False
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

    def _to_gpu(self):
        if _use_gpu:
            for attr in self.attrs:
                setattr(self, attr, cp.asarray(getattr(self, attr)))
        
    def _to_host(self):
        if _use_gpu:
            for attr in self.attrs:
                setattr(self, attr, getattr(self, attr).get())
        
    @property
    def Npart(self):
        return self.buffer_size - int(bool_sum(self._to_be_pruned))

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
        self.photon_delta = np.zeros(self.buffer_size)
        self.attrs += ["event", "photon_delta"]
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
        self.pair_delta = np.zeros(self.buffer_size)
        self.attrs += ["event", "pair_delta"]
        if _use_optical_depth:
            self.tau = np.zeros(self.buffer_size)
            self.attrs += ['tau']
        
    def _push_momentum(self, dt):
        if self.m > 0:
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
        if _use_gpu:
            from .gpu import tpb
            bpg = int(self.N_buffered / tpb) + 1
            fields.field_func[bpg, tpb](
                self.x, self.y, self.z, t, self.N_buffered, self._to_be_pruned,
                self.Ex, self.Ey, self.Ez, 
                self.Bx, self.By, self.Bz
            )
        else:
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
        N_photon = int(bool_sum(self.event))
        if N_photon == 0:
            return

        
        if hasattr(self, 'photon'):
            pho._extend(N_photon)

            # events are already false when marked as pruned in QED
            event_index = find_event_index(self.event)
            create_photon(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                pho.x, pho.y, pho.z, pho.ux, pho.uy, pho.uz,
                pho.inv_gamma, pho._to_be_pruned,
                event_index, self.photon_delta, pho.N_buffered, N_photon,
            )
            
            pho.N_buffered += N_photon
        

    def _create_pair(self, ele, pos):
        N_pair = int(bool_sum(self.event))
        if N_pair == 0:
            return
        
        
        if hasattr(self, 'bw_electron'):
            ele._extend(N_pair)
            pos._extend(N_pair)
            
            # events are already false when marked as pruned in QED
            event_index = find_event_index(self.event)
            create_pair(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                self._to_be_pruned,
                ele.x, ele.y, ele.z, ele.ux, ele.uy, ele.uz,
                ele.inv_gamma, ele._to_be_pruned,
                event_index, self.pair_delta, ele.N_buffered, N_pair,
                inverse_delta=False,
            )
            create_pair(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                self._to_be_pruned,
                pos.x, pos.y, pos.z, pos.ux, pos.uy, pos.uz,
                pos.inv_gamma, pos._to_be_pruned,
                event_index, self.pair_delta, pos.N_buffered, N_pair,
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
                setattr(self, attr, resize(getattr(self, attr), buffer_size_new))

            self._to_be_pruned[-(buffer_size_new-self.buffer_size):] = True


            # new buffer size
            self.buffer_size = buffer_size_new
            

    def _prune(self):
        '''
        call after copy to host if use gpu.
        '''
        selected = ~self._to_be_pruned
        for attr in self.attrs:
            setattr(self, attr, (getattr(self, attr))[selected])
        N = self.Npart
        self.buffer_size = N
        self.N_buffered = N