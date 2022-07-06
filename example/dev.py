from time import perf_counter, perf_counter_ns
from sfparticles import Particles

import numpy as np
from scipy.constants import pi, m_e, e, c

from sfparticles.simulation import simulate

um = 1e-6
fs = 1e-15

l0 = 0.8*um
omega0 = 2*pi*c / l0
k0 = omega0 / c

a0 = 1
w0 = 5*um
ctau = 10*um

a0_norm = e / (m_e * c * omega0)

gen = np.random.RandomState(0)
def init(N):
    x = gen.rand(N) * 1e-6
    y = gen.rand(N) * 1e-6
    z = gen.rand(N) * 1e-6
    ux = gen.rand(N) * 100
    uy = gen.rand(N) * 100
    uz = gen.rand(N) * 100
    return (x, y, z, ux, uy, uz)


def laser(x, y, z, t):
    E0 = a0 / a0_norm

    r2 = y**2 + z**2
    phi = k0*x - omega0*t

    Ex = 0.0
    Ey = E0 * np.sin(phi) * np.exp(-r2/w0**2) * np.exp(-phi**2 / (k0*ctau)**2)
    Ez = 0.0
    
    Bx = 0.0
    By = 0.0
    Bz = Ey / c
    return (Ex, Ey, Ez, Bx, By, Bz)

N = int(10000)
step = 1000
dt = 0.01*fs

photons = Particles('photon', 0, 0, 0)
electrons = Particles('electron', -1, 1, N, init(N), photon=photons)

tic = perf_counter_ns()
simulate(electrons, step=step, dt=dt, field_function=laser)
toc = perf_counter_ns()
print(f'{(toc - tic)/N/step:.2f} ns/step/particle')