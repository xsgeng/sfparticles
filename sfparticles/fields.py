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
        '''
        Construct Fields object from user function
        Parameters
        ----------
        field_func : Callable
            User defined function that takes 4 arguments: x, y, z, t
            and returns 6 values: Ex, Ey, Ez, Bx, By, Bz
        Returns
        -------
        Fields object
        '''

        # construct jitted function
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
        self.field_func_input = field_func

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
    '''
    Simple laser pulse without focusing.

    Parameters
    ----------
    a0 : float
        Amplitude of the laser pulse
    w0 : float
        w_0 of the laser pulse at 1/e
    ctau : float
        Width of the laser pulse in length at 1/e
    direction : int
        Direction of the laser pulse (+1 or -1)
    x0 : float
        center of the laser pulse
    wavelength : float
        Wavelength of the laser pulse
    pol_angle : float
        Polarization angle of the laser pulse in rad
    cep : float
        Carrier envelope phase of the laser pulse
    Returns
    -------
    Simple laser pulse Fields object
    '''
    assert direction == 1 or direction == -1, "direction must be +1 (+x) or -1 (-x)"

    omega0 = 2*pi*c / wavelength
    k0 = omega0 / c
    a0_norm = e / (m_e * c * omega0)
    E0 = a0 / a0_norm

    def _laser_pulse(x, y, z, t):
        x = direction * x
        r2 = y**2 + z**2
        phi = k0*(x - direction*x0 - c*t)

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
    '''
    Gaussian laser pulse

    Parameters
    ----------
    a0 : float
        Amplitude of the laser pulse
    w0 : float
        w_0 of the laser pulse at 1/e
    ctau : float
        Width of the laser pulse in length at 1/e
    direction : int
        Direction of the laser pulse (+1 or -1)
    x0 : float
        center of the laser pulse
    l0 : float
        Wavelength of the laser pulse
    pol_angle : float
        Polarization angle of the laser pulse in rad
    cep : float
        Carrier envelope phase of the laser pulse
    Returns
    -------
    gaussian pulse Fields object
    '''
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


def focused_gaussian_laser_pulse(a0, w0, ctau, direction=1, xfoc=0.0, x0=0.0, l0=0.8e-6, pol_angle=0.0, cep=0.0):
    '''
    Focused Gaussian laser pulse

    Ref. Electron Acceleration by a Tightly Focused Laser Beam [10.1103/PhysRevLett.88.095005]

    Parameters
    ----------
    a0 : float
        Amplitude of the laser pulse
    w0 : float
        w_0 of the laser pulse at 1/e
    ctau : float
        Width of the laser pulse in length at 1/e
    direction : int
        Direction of the laser pulse (+1 or -1)
    xfoc: float
        The x position of focal plane
    x0 : float
        center of the laser pulse
    l0 : float
        Wavelength of the laser pulse
    pol_angle : float
        Polarization angle of the laser pulse in rad
    cep : float
        Carrier envelope phase of the laser pulse
    Returns
    -------
    gaussian pulse Fields object
    '''
    assert direction == 1 or direction == -1, "direction must be +1 (+x) or -1 (-x)"

    omega0 = 2*pi*c/l0
    k0 = 2*pi/l0
    zR = pi*w0**2 / l0
    ϵ = w0/zR
    a0_norm = e / (m_e * c * omega0)
    E0 = a0 / a0_norm
    def _gaussian_pulse(x, y, z, t):
        x = direction * (x - xfoc)
        r2 = y**2 + z**2
        wx = w0 * sqrt(1 + (x/zR)**2)

        ψG  = atan(x/zR)
        ψR = k0*x * r2/2/(x**2 + zR**2)
        ψP = k0 * (x - direction * x0 - c*t)
        ψ = ψG + ψR + ψP + cep

        ρ = sqrt(r2) / w0
        ξ = y/w0
        ν = z/w0
        η = 1/sqrt(1 + (x/zR)**2) # w0/w


        ρ2 = ρ**2
        ρ4 = ρ**4
        ρ6 = ρ**6
        ρ8 = ρ**8
        E = E0 * w0/wx*exp(-r2/wx**2) * exp(-(ψ-cep)**2/(k0*ctau)**2)

        S0 = sin(ψ)
        S2 = η**2 *sin(ψ + 2*ψG)
        S3 = η**3 *sin(ψ + 3*ψG)
        S4 = η**4 *sin(ψ + 4*ψG)
        S5 = η**5 *sin(ψ + 5*ψG)
        S6 = η**6 *sin(ψ + 6*ψG)

        C1 = η**1 *cos(ψ + 1*ψG)
        C2 = η**2 *cos(ψ + 2*ψG)
        C3 = η**3 *cos(ψ + 3*ψG)
        C4 = η**4 *cos(ψ + 4*ψG)
        C5 = η**5 *cos(ψ + 5*ψG)
        C6 = η**6 *cos(ψ + 6*ψG)
        C7 = η**7 *cos(ψ + 7*ψG)

        Ex_ = E * ξ*(ϵ*C1 + ϵ**3*(-C2/2 + ρ2*C3 - ρ4*C4/4) + ϵ**5*(-3/8*C3 - 3/8*ρ2*C4 + 17/16*ρ4*C5 - 3/8*ρ6*C6 + ρ8*C7/32))
        Ey_ = E * (S0 + ϵ**2*(ξ**2*S2 - ρ4*S3/4) + ϵ**4*(S2/8-ρ2*S3/4-ρ2*(ρ2-16*ξ**2)*S4/16 - ρ4*(ρ2+2*ξ**2)*S5/8 + ρ8*S6/32))
        Ez_ = E * ξ*ν*(ϵ**2*S2 + ϵ**4*(ρ2*S4 - ρ4*S5/4))

        Bx_ = E/c * ν*(ϵ*C1 + ϵ**3*(C2/2 + ρ2*C3/2 - ρ4*C4/4) + ϵ**5*(3/8*C3 + 3/8*ρ2*C4 + 3/16*ρ4*C5 - ρ6*C6/4 + ρ8*C7/32))
        By_ = 0
        Bz_ = E/c * (S0 + ϵ**2*(ρ2*S2/2 - ρ4*S3/4) + ϵ**4*(-S2/8 + ρ2*S3/4 + 5/16*ρ4*S4 - ρ6*S5/4 + ρ8*S6/32))
        
        Ex = Ex_
        Ey = Ey_ * cos(pol_angle) - Ez_ * sin(pol_angle)
        Ez = Ey_ * sin(pol_angle) + Ez_ * cos(pol_angle)
        
        Bx = Bx_
        By = (By_ * cos(pol_angle) - Bz_ * sin(pol_angle)) * direction
        Bz = (By_ * sin(pol_angle) + Bz_ * cos(pol_angle)) * direction
        return (Ex, Ey, Ez, Bx, By, Bz)

    return Fields(_gaussian_pulse)


def static_field(Ex=0, Ey=0, Ez=0, Bx=0, By=0, Bz=0):
    def _static_field(x, y, z, t):
        return (Ex, Ey, Ez, Bx, By, Bz)
    return Fields(_static_field)