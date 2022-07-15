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
ux0 = 10000
def init(N):
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    ux = np.full(N, ux0)
    uy = np.zeros(N)
    uz = np.zeros(N)
    return (x, y, z, ux, uy, uz)

laser1 = simple_laser_pulse(a0, w0, ctau)
laser2 = simple_laser_pulse(a0, w0, ctau, pol_angle=pi/2, cep=pi/2)

laser = laser1 + laser2

Bz0 = 1E6
Bz = static_field(Bz=Bz0)

R = m_e*c*ux0 / e / Bz0
T = 2*pi*R / c

N = int(10000)
step = 2000
dt = T / 4 / step

photons = Particles('photon', 0, 0)
positrons = Particles('positron', q=1, m=1)
electrons = Particles('electron', q=-1, m=1, N=N, props=init(N))

photons.set_pair(electrons, positrons)
electrons.set_photon(photons)
positrons.set_photon(photons)

simulate(electrons, positrons, photons, step=step, dt=dt, fields=Bz)

print(positrons.x.shape[0], photons.x.shape[0], photons.chi.max())

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.hist2d(
    electrons.x/1E-6, electrons.y/1E-6, bins=1024, range=[[-R/1E-6, R/1E-6], [0, 2*R/1E-6]]
)

fig.savefig('test.png', dpi=300)

