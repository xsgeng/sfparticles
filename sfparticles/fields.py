from numba import njit, prange, vectorize

class Fields(object):
    def __init__(self, field_func) -> None:
        field_func_numba = njit(field_func, cache=True)
        
        @njit(parallel=True, cache=True)
        def field_function(x, y, z, t, N, Ex, Ey, Ez, Bx, By, Bz):
            for ip in prange(N):
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip] = field_func_numba(x[ip], y[ip], z[ip], t)

        self.field_func = field_function


    def _eval_field(self, particles, t):
        self.field_func(
            particles.x, particles.y, particles.z, t, particles.N, 
            particles.Ex, particles.Ey, particles.Ez, 
            particles.Bx, particles.By, particles.Bz
        )


def laser_pulse(a0, w0, ctau, x0, zf):
    def _laser_pulse(x, y, z, t):
        pass
    return _laser_pulse