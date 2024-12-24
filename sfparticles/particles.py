from enum import Enum, auto
from typing import Tuple, Union
from weakref import ref
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
    from .gpu import push_position, boris, boris_tbmt, LL_push, CLL_push, update_chi, \
        pick_hard_photon, photon_recoil, create_pair, create_photon, \
        find_event_index, bool_sum
else:
    from numpy import resize
    from .cpu import push_position, boris, boris_tbmt, LL_push, CLL_push, update_chi, \
        pick_hard_photon, photon_recoil, create_pair, create_photon, \
        find_event_index, bool_sum

class RadiationReactionType(Enum):
    """
    Radiation reaction type enumeration.
    
    Values:
        NONE: No radiation reaction
        PHOTON: Radiation reaction through photon emission
        LL: Approximated Landau-Lifshitz equation for gamma >> 1
        CLL: Quantum-corrected Landau-Lifshitz equation
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

        q : int
            Charge in units of electron charge (±1 for electrons/positrons)
        m : float
            Mass in units of electron mass
        N : int
            Number of particles
        props : Tuple(x, y, z, ux, uy, uz, [sx, sy, sz])
            Initial particle state, can include spin vectors.
            These property vectors have length equal to the buffer length,
            with the first N elements being active particle properties.
        RR : RadiationReactionType
            Radiation reaction type. Invalid for photons (m=0).
            NONE: No radiation reaction
            PHOTON: Radiation reaction through photon emission
            LL: Landau-Lifshitz equation
            CLL: Quantum-corrected Landau-Lifshitz equation
        push : bool, default=True
            Whether to simulate particle motion
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

    def _to_gpu(self) -> None:
        """
        Transfer particle data to GPU memory.
        Only executed if GPU support is enabled.
        """
        if _use_gpu:
            for attr in self.attrs:
                setattr(self, attr, cp.asarray(getattr(self, attr)))
        
    def _to_host(self) -> None:
        """
        Transfer particle data from GPU back to host memory.
        Only executed if GPU support is enabled.
        """
        if _use_gpu:
            for attr in self.attrs:
                setattr(self, attr, getattr(self, attr).get())
        
    @property
    def Npart(self) -> int:
        """
        Get the current number of active particles.
        
        Returns:
            Number of particles that are not marked for pruning
        """
        return self.N_buffered - int(bool_sum(self._to_be_pruned, self.N_buffered))

    @property
    def gamma(self) -> float:
        """
        Get the relativistic gamma factor.
        
        Returns:
            Gamma factor (1/sqrt(1-v^2/c^2)) for massive particles
            or energy/mc^2 for photons
        """
        return 1.0 / self.inv_gamma

    def set_photon(self, photon: 'Particles') -> None:
        """
        Configure particle for photon emission.
        
        Sets up the necessary arrays and references for tracking
        photon emission events from this particle.
        
        Args:
            photon: Particles instance that will store emitted photons
            
        Raises:
            AssertionError: If particle is massless or using LL/CLL radiation
            reaction, or if photon particle properties are invalid
        """
        assert self.m > 0, 'photon cannot radiate photon'
        assert self.RR != RadiationReactionType.LL and self.RR != RadiationReactionType.CLL, 'LL equation does not radiate photon'
        assert isinstance(photon, Particles), 'photon must be Particle class'
        assert photon.m == 0 and photon.q == 0, 'photon must be m=0 and q=0'
        self.radiating = True
        self.photon_ref = ref(photon) # avoid gc of mutual reference
        self.event = np.full(self.buffer_size, False)
        self.photon_delta = np.zeros(self.buffer_size)
        self.attrs += ["event", "photon_delta"]
        if _use_optical_depth:
            self.tau = np.zeros(self.buffer_size)
            self.attrs += ['tau']
        
        
    def set_pair(self, electron: 'Particles', positron: 'Particles') -> None:
        """
        Configure photon for electron-positron pair production.
        
        Sets up the necessary arrays and references for tracking
        pair production events from this photon.
        
        Args:
            electron: Particles instance that will store produced electrons
            positron: Particles instance that will store produced positrons
            
        Raises:
            AssertionError: If particle is massive or if electron/positron
            properties are invalid
        """
        assert self.m == 0, 'massive particle cannot create BW pair'
        assert isinstance(electron, Particles) and isinstance(positron, Particles), 'pair must be tuple or list of Particle class'
        assert electron.m == m_e and electron.q == -e, f'first of the pair must be electron'
        assert positron.m == m_e and positron.q ==  e, f'second of the pair must be positron'
        self.bw = True
        self.bw_electron_ref = ref(electron) # avoid gc of mutual reference
        self.bw_positron_ref = ref(positron)
        self.event = np.full(self.buffer_size, False)
        self.pair_delta = np.zeros(self.buffer_size)
        self.attrs += ["event", "pair_delta"]
        if _use_optical_depth:
            self.tau = np.zeros(self.buffer_size)
            self.attrs += ['tau']
        
    def _push_momentum(self, dt: float) -> None:
        """
        Update particle momentum using the Boris pusher algorithm.
        
        For massive particles, applies the Boris algorithm and optionally
        radiation reaction effects based on the RR type.
        
        Args:
            dt: Time step in seconds
        """
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
        if self.RR == RadiationReactionType.CLL:
            # LL push uses chi value
            # see LL_push_inline for details
            self._calculate_chi()
            CLL_push(
                self.ux, self.uy, self.uz, 
                self.inv_gamma, self.chi,  
                self.N_buffered, self._to_be_pruned, dt
            )


    def _push_position(self, dt: float) -> None:
        """
        Update particle positions based on their momenta.
        
        Args:
            dt: Time step in seconds
        """
        push_position(
            self.x, self.y, self.z, 
            self.ux, self.uy, self.uz, 
            self.inv_gamma, 
            self.N_buffered, self._to_be_pruned, dt
        )


    def _eval_field(self, fields: Fields, t: float) -> None:
        """
        Evaluate electromagnetic fields at particle positions.
        
        Uses the provided Fields object to calculate E and B fields
        at each particle's position, handling both CPU and GPU implementations.
        
        Args:
            fields: Fields object containing field calculation functions
            t: Current simulation time in seconds
        """
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


    def _calculate_chi(self) -> None:
        """
        Calculate quantum parameter chi for each particle.
        
        Updates the chi array using current particle momenta
        and electromagnetic fields. Chi represents the quantum
        nonlinearity parameter that determines QED effects.
        """
        update_chi(
            self.Ex, self.Ey, self.Ez, 
            self.Bx, self.By, self.Bz, 
            self.ux, self.uy, self.uz,
            self.inv_gamma, self.chi, self.N_buffered, self._to_be_pruned, 
        )


    def _photon_event(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate photon emission events.
        
        Uses either optical depth tables or rejection sampling
        to determine photon emission events and energies.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Tuple containing:
            - Boolean array of emission events
            - Array of photon energy fractions
        """
        if _use_optical_depth:
            update_tau_e(self.tau, self.inv_gamma, self.chi, dt, self.N_buffered, self._to_be_pruned, self.event, self.photon_delta)
        else:
            photon_from_rejection_sampling(self.inv_gamma, self.chi, dt, self.N_buffered, self._to_be_pruned, self.event, self.photon_delta )
        return self.event, self.photon_delta

    
    def _pair_event(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate pair production events.
        
        Uses either optical depth tables or rejection sampling
        to determine pair production events and energy sharing.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Tuple containing:
            - Boolean array of pair production events
            - Array of energy sharing fractions
        """
        if _use_optical_depth:
            update_tau_gamma(self.tau, self.inv_gamma, self.chi, dt, self.N_buffered, self._to_be_pruned, self.event, self.pair_delta)
        else:
            pair_from_rejection_sampling(self.inv_gamma, self.chi, dt, self.N_buffered, self._to_be_pruned, self.event, self.pair_delta )
        return self.event, self.pair_delta

    
    def _pick_hard_photon(self, threshold: float = 2.0) -> None:
        """
        Filter photon emission events based on energy threshold.
        
        Only keeps emission events where the photon energy exceeds
        the threshold times the particle energy.
        
        Args:
            threshold: Minimum ratio of photon to particle energy
        """
        pick_hard_photon(self.event, self.photon_delta, self.inv_gamma, threshold, self.N_buffered)
        
        
    def _create_photon(self, pho: 'Particles') -> None:
        """
        Create new photon particles from emission events.
        
        For each emission event, creates a new photon with appropriate
        energy and momentum, and applies recoil to the emitting particle
        if radiation reaction is enabled.
        
        Args:
            pho: Particles instance to store the created photons
        """
        N_photon = int(bool_sum(self.event, self.N_buffered))
        if N_photon == 0:
            return

        
        if hasattr(self, 'photon_ref'):
            # events are already false when marked as pruned in QED
            event_index = find_event_index(self.event, N_photon, self.N_buffered)
            pho._extend(N_photon)
            
            create_photon(
                self.x, self.y, self.z, self.ux, self.uy, self.uz,
                pho.x, pho.y, pho.z, pho.ux, pho.uy, pho.uz,
                pho.inv_gamma, pho._to_be_pruned,
                event_index, self.photon_delta, pho.N_buffered, N_photon,
            )
            
            pho.N_buffered += N_photon
        
        # RR
        if self.RR == RadiationReactionType.PHOTON:
            photon_recoil(self.ux, self.uy, self.uz, self.inv_gamma, self.event, self.photon_delta, self.N_buffered, self._to_be_pruned)

    def _create_pair(self, ele: 'Particles', pos: 'Particles') -> None:
        """
        Create electron-positron pairs from pair production events.
        
        For each pair production event, creates an electron and positron
        with appropriate energy sharing and momentum conservation.
        
        Args:
            ele: Particles instance to store created electrons
            pos: Particles instance to store created positrons
        """
        N_pair = int(bool_sum(self.event, self.N_buffered))
        if N_pair == 0:
            return
        
        
        if hasattr(self, 'bw_electron_ref'):
            # events are already false when marked as pruned in QED and extend methods
            event_index = find_event_index(self.event, N_pair, self.N_buffered)
            
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
            

    def _extend(self, N_new: int) -> None:
        """
        Extend particle arrays to accommodate new particles.
        
        Increases the buffer size if needed and initializes new
        array elements appropriately. Growth strategy adds 25%
        plus requested new particles.
        
        Args:
            N_new: Number of new particle slots needed
        """
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
            

    def _prune(self) -> None:
        """
        Remove deleted particles and compact arrays.
        
        Filters out particles marked for deletion and resizes
        arrays to contain only active particles. Must be called
        after copying data from GPU if GPU support is enabled.
        """
        selected = ~self._to_be_pruned
        for attr in self.attrs:
            setattr(self, attr, (getattr(self, attr))[selected])
        N = selected.sum()
        self.buffer_size = N
        self.N_buffered = N

class SpinParticles(Particles):
    """
    Particle class that includes spin dynamics.
    
    Extends the base Particles class to include spin vectors and
    spin-dependent equations of motion using the 
    Boris-Thomas-BMT algorithm.
    
    Attributes:
        sx, sy, sz: Spin vector components
        ae: Anomalous magnetic moment (g-2)/2
    """
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
        """
        Initialize a SpinParticles object.
        
        In addition to the base Particles parameters, includes
        spin vectors and anomalous magnetic moment.
        
        Args:
            name: Particle name identifier
            q: Charge in units of electron charge
            m: Mass in units of electron mass
            N: Number of particles
            props: Tuple of 9 components (x,y,z, ux,uy,uz, sx,sy,sz)
            RR: Radiation reaction type
            ae: Anomalous magnetic moment, defaults to electron value
            push: Whether to simulate particle motion
        """
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


    def _push_momentum(self, dt: float) -> None:
        """
        Update particle momentum including spin effects.
        
        Uses the Boris-Thomas-BMT algorithm to evolve both
        momentum and spin vectors in electromagnetic fields.
        
        Args:
            dt: Time step in seconds
        """
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

def prepare_props(props: Tuple, N: int) -> list:
    """
    Prepare particle property arrays from input data.
    
    Converts various input formats (arrays, scalars, lists) into
    properly sized numpy arrays for particle properties.
    
    Args:
        props: Tuple of property values in various formats
        N: Number of particles
        
    Returns:
        List of numpy arrays containing particle properties
        
    Raises:
        AssertionError: If property dimensions don't match N
    """
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
