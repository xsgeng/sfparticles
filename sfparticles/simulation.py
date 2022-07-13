from time import perf_counter_ns
from typing import Callable, List

from .fields import Fields
from .particles import Particles, c

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
            particles._calculate_chi()

            if hasattr(particles, 'pair'):
                particles._pair_event(dt)
            if hasattr(particles, 'photon'):
                particles._photon_event(dt)
        # create particles
        # seperated from events generation
        # since particles created in the current loop do NOT further create particle
        for particles in all_particles:
            if hasattr(particles, 'photon_delta'):
                particles._create_photon()
            if hasattr(particles, 'pair_delta'):
                particles._create_pair()
        
        t += dt
        if (istep+1) % 100 == 0 :
            toc = perf_counter_ns()
            Ntotal = sum([particles.Npart for particles in all_particles])
            print(
                f'step: {istep+1}, c*time: {c*t/1e-6:.2f} um, ',
                ', '.join([f"{particles.Npart} {particles.name}" for particles in all_particles]),
                f', {(toc-tic)/100/Ntotal:.2f} ns/particle'
            )
            tic = perf_counter_ns()

    for particles in all_particles:
        particles._prune()
