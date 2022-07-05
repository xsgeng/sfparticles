from typing import Tuple, Union
import numpy as np
from scipy.constants import c
from numba import njit, prange

class Particles(object):
    def __init__(
        self,
        q: float, m: float, N: int,
        props : Tuple = None,
        has_spin = False,
        photon = None, pair = None,
    ) -> None:
        self.q = q
        self.m = m
        self.N = N
        self.has_spin = has_spin

        assert m < 0, 'negative mass'
        if m > 0:
            assert pair is None, 'massive particle cannot create BW pair'
            self.photon = photon
        if m == 0:
            assert photon is None, 'photon cannot radiate photon'
            self.pair = pair

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
                assert len(prop.shape) == 1, 'given property is not vector'
                assert prop.shape[0] == N, 'given N does not match given property length'

            x, y, z, ux, uy, uz = props[:6]
            if len(props) == 9 and has_spin:
                sx, sy, sz = props[7:]

        # position, momentum and spin vectors
        self.x = x
        self.y = y
        self.z = z
        self.ux = ux
        self.uy = uy
        self.uz = uz

        if self.has_spin:
            self.sx = sx
            self.sy = sy
            self.sz = sz

        # gamma factor
        if m > 0:
            self.inv_gamma = 1./np.sqrt( 1 + ux**2 + uy**2 + uz**2 )
        else:
            self.inv_gamma = 1./np.sqrt( ux**2 + uy**2 + uz**2 )
        
        # quantum parameter
        self.chi = np.zeros(N)

        # fields at particle positions
        self.Ez = np.zeros(N)
        self.Ex = np.zeros(N)
        self.Ey = np.zeros(N)
        self.Bz = np.zeros(N)
        self.Bx = np.zeros(N)
        self.By = np.zeros(N)


    # def set_field(self, field_function, use_cuda=False):
    #     self.fieldd_function = field_function
    #     self.use_cuda = use_cuda


    # def _eval_field(self):
    #     pass


    def _push_momentum(self, dt):
        pass

    def _push_position(self, dt):
        push_position(
            self.x, self.y, self.z, 
            self.ux, self.uy, self.uz, 
            self.inv_gamma, 
            self.N, dt
        )

    def _radiate_photons(self, dt):
        pass


    def _create_pair(self, dt):
        pass


@njit(parallel=True, cache=True)
def push_position( x, y, z, ux, uy, uz, inv_gamma, N, dt ):
    """
    Advance the particles' positions over `dt` using the momenta `ux`, `uy`, `uz`,
    """
    # Timestep, multiplied by c
    cdt = c*dt

    # Particle push (in parallel if threading is installed)
    for ip in prange(N) :
        x[ip] += cdt * inv_gamma[ip] * ux[ip]
        y[ip] += cdt * inv_gamma[ip] * uy[ip]
        z[ip] += cdt * inv_gamma[ip] * uz[ip]


# @njit(parallel=True, cache=True)
# def boris( ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, q, m, N, dt ) :
#     """
#     Advance the particles' momenta, using numba
#     """
#     # Set a few constants
#     econst = q*dt/(m*c)
#     bconst = 0.5*q*dt/m

#     # Loop over the particles (in parallel if threading is installed)
#     for ip in prange(N) :
#         ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
#             ux[ip], uy[ip], uz[ip], inv_gamma[ip],
#             Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst )

#     return ux, uy, uz, inv_gamma