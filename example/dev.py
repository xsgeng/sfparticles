from time import perf_counter
from sfparticles.particles import Particles

import numpy as np
from scipy.constants import m_e, e

gen = np.random.RandomState(0)
def init(N):
    x = gen.rand(N) * 1e-6
    y = gen.rand(N) * 1e-6
    z = gen.rand(N) * 1e-6
    ux = gen.rand(N) * 100
    uy = gen.rand(N) * 100
    uz = gen.rand(N) * 100
    return (x, y, z, ux, uy, uz)


electrons = Particles(-e, m_e, 1000000, init(1000000))


tic = perf_counter()
for i in range(1000):
    electrons._push_position(1e-15)
toc = perf_counter()
print(toc - tic, ' s')
# photons = Particles(0, 0, 0)