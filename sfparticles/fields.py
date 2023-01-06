from typing import Callable
from numba import njit, prange
from numba.core.registry import CPUDispatcher
from scipy.constants import c, pi, e, m_e
from math import sin, cos, exp, atan, sqrt

from .gpu import _use_gpu
if _use_gpu:
    from numba import cuda
    from numba.cuda.dispatcher import CUDADispatcher
class Fields(object):
    def __init__(self, field_func : Callable) -> None:
        
        if _use_gpu:
            if isinstance(field_func, CUDADispatcher):
                field_func_inline = field_func
            else:
                field_func_inline = cuda.jit(field_func)
            @cuda.jit
            def field_function(x, y, z, t, N, to_be_pruned, Ex, Ey, Ez, Bx, By, Bz):
                ip = cuda.grid(1)
                if ip < N and ~to_be_pruned[ip]:
                    Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip] = field_func_inline(x[ip], y[ip], z[ip], t)
        else:
            if isinstance(field_func, CPUDispatcher):
                field_func_inline = field_func
            else:
                field_func_inline = njit(field_func, cache=False)
            @njit(parallel=True, cache=False)
            def field_function(x, y, z, t, N, to_be_pruned, Ex, Ey, Ez, Bx, By, Bz):
                for ip in prange(N):
                    if to_be_pruned[ip]:
                        continue
                    Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip] = field_func_inline(x[ip], y[ip], z[ip], t)

        self.field_func_inline = field_func_inline
        self.field_func = field_function

    def __add__(self, other):
        field_func_inline1 = self.field_func_inline
        field_func_inline2 = other.field_func_inline

        def out(x, y, z, t):
            Ex1, Ey1, Ez1, Bx1, By1, Bz1 = field_func_inline1(x, y, z, t)
            Ex2, Ey2, Ez2, Bx2, By2, Bz2 = field_func_inline2(x, y, z, t)
            return (Ex1+Ex2, Ey1+Ey2, Ez1+Ez2, Bx1+Bx2, By1+By2, Bz1+Bz2)

        return Fields(out)



# TODO
# def check_field_function():
#     pass

def simple_laser_pulse(a0, w0, ctau, direction=1,  x0=0, wavelength=0.8e-6, pol_angle=0, cep=0):
    assert direction == 1 or direction == -1, "direction must be +1 (+x) or -1 (-x)"

    omega0 = 2*pi*c / wavelength
    k0 = omega0 / c
    a0_norm = e / (m_e * c * omega0)
    E0 = a0 / a0_norm

    def _laser_pulse(x, y, z, t):
        r2 = y**2 + z**2
        phi = k0*(x-x0 - direction*c*t)

        E = E0 * cos(phi + cep) * exp(-r2/w0**2) * exp(-phi**2 / (k0*ctau)**2)
        Ex = 0.0
        Ey = E * cos(pol_angle)
        Ez = E * sin(pol_angle)
        
        Bx = 0.0
        By = -Ez / c * direction
        Bz = Ey / c * direction

        return (Ex, Ey, Ez, Bx, By, Bz)

    return Fields(_laser_pulse)

def gaussian_laser_pulse(a0, w0, ctau, direction=1, x0=0.0, l0=0.8e-6, pol_angle=0.0, cep=0.0):
    assert direction == 1 or direction == -1, "direction must be +1 (+x) or -1 (-x)"

    omega0 = 2*pi*c/l0
    k0 = 2*pi/l0
    zR = pi*w0**2 / l0
    a0_norm = e / (m_e * c * omega0)
    def _gaussian_pulse(x, y, z, t):
        r2 = y**2 + z**2
        wx = w0 * sqrt(1 + (x/zR)**2)
        Rx = x * (1 + (zR/x)**2)
        gouy = atan(x/zR)
        
        E0 = a0 / a0_norm
        phi = k0 * (x - x0 - direction*c*t) + k0*r2/2/Rx - gouy
        E = E0 * cos(phi+cep) * w0/wx * exp(-r2/wx**2) * exp(-phi**2/(k0*ctau)**2)
        Ex = 0.0
        Ey = E * cos(pol_angle)
        Ez = E * sin(pol_angle)
        
        Bx = 0.0
        By = -Ez / c * direction
        Bz = Ey / c * direction
        return (Ex, Ey, Ez, Bx, By, Bz)

    return Fields(_gaussian_pulse)


def static_field(Ex=0, Ey=0, Ez=0, Bx=0, By=0, Bz=0):
    def _static_field(x, y, z, t):
        return (Ex, Ey, Ez, Bx, By, Bz)
    return Fields(_static_field)