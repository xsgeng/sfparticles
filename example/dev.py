from time import perf_counter
from sfparticles import Particles

import numpy as np
from scipy.constants import m_e, e

from sfparticles.simulation import simulate

gen = np.random.RandomState(0)
def init(N):
    x = gen.rand(N) * 1e-6
    y = gen.rand(N) * 1e-6
    z = gen.rand(N) * 1e-6
    ux = gen.rand(N) * 100
    uy = gen.rand(N) * 100
    uz = gen.rand(N) * 100
    return (x, y, z, ux, uy, uz)

N = int(1E5)
photons = Particles('photon', 0, 0, 0)
electrons = Particles('electron', -1, 1, N, init(N), photon=photons)

print(electrons is electrons)
tic = perf_counter()
simulate(electrons, step=1000, dt=1E-15)
toc = perf_counter()
print(toc - tic, ' s')