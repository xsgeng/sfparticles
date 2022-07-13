from sfparticles import Particles
import matplotlib.pyplot as plt

import numpy as np
from scipy.constants import pi, m_e, e, c
from sfparticles.fields import simple_laser_pulse

from sfparticles import Simulation

um = 1e-6
fs = 1e-15

l0 = 0.8*um
omega0 = 2*pi*c / l0
k0 = omega0 / c

a0 = 1000.0
w0 = 5*um
ctau = 10*um

a0_norm = e / (m_e * c * omega0)

N = int(1)
Tsim = 6*ctau/c
dt = 1*um / c / 100
step = int(Tsim / dt)

gen = np.random.RandomState(0)
def init(N):
    x = gen.randn(N) * 0.1*um
    y = gen.randn(N) * 0.1*um
    z = gen.randn(N) * 0.1*um
    ux = np.zeros(N)
    uy = np.zeros(N)
    uz = np.zeros(N)
    return (x, y, z, ux, uy, uz)

laser1 = simple_laser_pulse(a0, w0, ctau, direction=-1, x0=3*ctau)
laser2 = simple_laser_pulse(a0, w0, ctau, direction=1, x0=-3*ctau)

laser = laser1 + laser2


photons = Particles('photon', 0, 0)
positrons = Particles('positron', q=1, m=1)
electrons = Particles('electron', q=-1, m=1, N=N, props=init(N))

photons.set_pair((electrons, positrons))
electrons.set_photon(photons)
positrons.set_photon(photons)

sim = Simulation(electrons, positrons, photons, dt=dt, fields=laser)
sim.start(step)
print(positrons.Npart)

fig, ax = plt.subplots()

ax.hist2d(
    positrons.x/1E-6, positrons.y/1E-6, 
    bins=128, 
)

fig.savefig('test.png', dpi=300)

