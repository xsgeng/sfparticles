from time import perf_counter_ns
from typing import Callable, List

from .fields import Fields
from .particles import Particles

from numba import njit, prange, get_num_threads

# TODO:
# class Simulation(object):
#     def __init__(self) -> None:
#         pass
#     def add_particles(self, particles):
#         pass
#     def add_field(self, field):
#         pass
#     def add_diag_callback(self, diag_func):
#         pass
#     def start(self, nstep):
#         pass


def simulate(
    *all_particles : Particles,
    step: int,
    dt: float,
    fields : Fields,
):

    t = 0.0
    tic = perf_counter_ns()

    for istep in range(step):
        # push particles
        for particles in all_particles:
            particles._eval_field(fields, t)

            particles._push_position(0.5*dt)
            particles._push_momentum(dt)
            particles._push_position(0.5*dt)

        # QED
        for particles in all_particles:
            particles._eval_field(fields, t)
            particles._calculate_chi()

            if hasattr(particles, 'pair'):
                particles._create_pair(dt)
            if hasattr(particles, 'photon'):
                particles._radiate_photons(dt)
        
        t += dt

    toc = perf_counter_ns()
    print(f'{(toc - tic)/all_particles[0].N_buffered/step:.2f} ns/step/particle')

    for particles in all_particles:
        particles._prune()
