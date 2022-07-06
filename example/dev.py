from time import perf_counter, perf_counter_ns
from sfparticles import Particles

import numpy as np
from scipy.constants import pi, m_e, e, c
from numba import njit, prange
from sfparticles.fields import simple_laser_pulse, static_field

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

laser1 = simple_laser_pulse(a0, w0, ctau)
laser2 = simple_laser_pulse(a0, w0, ctau, pol_angle=pi/2, cep=pi/2)

laser = laser1 + laser2

By = static_field(By=1E6)

N = int(10000)
step = 10000
dt = 0.01*fs

photons = Particles('photon', 0, 0, 0)
electrons = Particles('electron', -1, 1, N, init(N), photon=photons)

simulate(electrons, step=step, dt=dt, field=By)
