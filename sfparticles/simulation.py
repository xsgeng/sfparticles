from time import perf_counter_ns
from typing import Callable, List

from .fields import Fields
from .particles import Particles

from numba import njit, prange


def simulate(
    *all_particles : Particles,
    step: int,
    dt: float,
    field : Fields = None,
):

    t = 0.0
    tic = perf_counter_ns()

    for istep in range(step):
        # push particles
        for particles in all_particles:
            if field:
                field._eval_field(particles, t)

            particles._push_position(0.5*dt)
            particles._push_momentum(dt)
            particles._push_position(0.5*dt)

        # QED
        for particles in all_particles:
            particles._calculate_chi()
            if particles.photon:
                photons = particles._radiate_photons(dt)
            if particles.pair:
                bw_pair = particles._create_pair(dt)
        t += dt*istep

    toc = perf_counter_ns()
    print(f'{(toc - tic)/all_particles[0].N/step:.2f} ns/step/particle')
