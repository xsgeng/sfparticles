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
    from .gpu import push_position, boris, boris_tbmt, LL_push, update_chi, \
        pick_hard_photon, photon_recoil, create_pair, create_photon, \
        find_event_index, bool_sum
else:
    from numpy import resize
    from .cpu import push_position, boris, boris_tbmt, LL_push, update_chi, \
        pick_hard_photon, photon_recoil, create_pair, create_photon, \
        find_event_index, bool_sum

class RadiationReactionType(Enum):
    """
    辐射反作用类型
    `None` : 无辐射反作用
    `photon` : 光子产生辐射反作用
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
        push = True,
    ) -> None:
        """
        ## initialize particles

        Initializes a particle object with the given properties.

        parameters
        ----------

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
            self.x = np.zeros(N)
            self.y = np.zeros(N)
            self.z = np.zeros(N)
            self.ux = np.zeros(N)
            self.uy = np.zeros(N)
            self.uz = np.zeros(N)
                

        if props:
            assert len(props) == 6, 'given properties must has length of 6'

            props_ = prepare_props(props, N)

            self.x, self.y, self.z, self.ux, self.uy, self.uz = props_

        # position, momentum and spin vectors
        self.attrs += ['x', 'y', 'z', 'ux', 'uy', 'uz']


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
        '''
        send data to gpu
        '''
        if _use_gpu:
            for attr in self.attrs:
                setattr(self, attr, cp.asarray(getattr(self, attr)))
        
    def _to_host(self):
        '''
        send data to host
        '''
        if _use_gpu:
            for attr in self.attrs:
                setattr(self, attr, getattr(self, attr).get())
        
    @property
    def Npart(self):
        '''
        Count number of particles
        '''
        return self.buffer_size - int(bool_sum(self._to_be_pruned))

    @property
    def gamma(self):
        '''
        Photon gamma factor
        '''
        return 1.0 / self.inv_gamma

    def set_photon(self, photon):
        '''
        set photon properties
        '''
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
        '''
        Set pair properties
        '''
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
        '''
        Push particle momentum
        '''
        if self.m == 0:
            return
        boris(
            self.ux, self.uy, self.uz,
            self.inv_gamma,
            self.Ex, self.Ey, self.Ez, 
            self.Bx, self.By, self.Bz,
            self.q, self.N_buffered, self._to_be_pruned, dt
        )
           
        if self.RR == RadiationReactionType.LL:
            # LL push uses chi value
            # see LL_push_inline for details
            self._calculate_chi()
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
        '''
        Evaluate EM fields at particle position
        '''
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
            # events are already false when marked as pruned in QED
            event_index = find_event_index(self.event, N_photon)
            pho._extend(N_photon)
            
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
            # events are already false when marked as pruned in QED and extend methods
            event_index = find_event_index(self.event, N_pair)
            
            ele._extend(N_pair)
            pos._extend(N_pair)
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
                if attr == 'event':
                    self.event[-(buffer_size_new-self.buffer_size):] = False

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
        N = selected.sum()
        self.buffer_size = N
        self.N_buffered = N

class SpinParticles(Particles):
    def __init__(
        self, 
        name: str, 
        q: int, 
        m: float, 
        N: int = 0, 
        props: Tuple = None, 
        RR: RadiationReactionType = RadiationReactionType.PHOTON, 
        ae: float = 1.14e-3, 
        push: bool = True
    ) -> None:
        if props is None:
            if m == 0:
                assert N == 0, "cannot initialize photons with only N without props."
            super().__init__(name, q, m, N, props, RR, push)
            self.sx = np.zeros(N)
            self.sy = np.zeros(N)
            self.sz = np.zeros(N)
        if props:
            assert len(props) == 9, 'given properties must has length of 9'
            super().__init__(name, q, m, N, props[:6], RR, ae, push)
            props_ = prepare_props(props[6:])

            self.sx, self.sy, self.sz = props_

        self.ae = ae
        self.attrs += ['sx', 'sy', 'sz']


    def _push_momentum(self, dt):
        '''
        Push particle momentum
        '''
        if self.m == 0:
            return
        boris_tbmt(
            self.ux, self.uy, self.uz,
            self.inv_gamma,
            self.sx, self.sy, self.sz,
            self.Ex, self.Ey, self.Ez, 
            self.Bx, self.By, self.Bz,
            self.q, self.ae, self.N_buffered, self._to_be_pruned, dt
        )
           
        if self.RR == RadiationReactionType.LL:
            # LL push uses chi value
            # see LL_push_inline for details
            self._calculate_chi()
            LL_push(
                self.ux, self.uy, self.uz, 
                self.inv_gamma, self.chi,  
                self.N_buffered, self._to_be_pruned, dt
            )

def prepare_props(props, N):
    props_ = []
    for prop in props:
        if isinstance(prop, np.ndarray):
            assert len(prop.shape) == 1, 'given property is not vector'
            assert prop.shape[0] == N, 'given N does not match given property length'
            props_.append(prop)
        if isinstance(prop, (int, float)):
            assert N == 1, 'given N does not match given property length'
            props_.append(np.asarray([prop], dtype=np.float64))
        if isinstance(prop, (list, tuple)):
            assert len(prop) == N, 'given N does not match given property length'
            props_.append(np.asarray(prop, dtype=np.float64))
    return props_