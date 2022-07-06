from typing import Callable, List
from .particles import Particles


def simulate(
    *all_particles : Particles,
    step: int,
    dt: float,
    field_function : Callable = None,
):
    for istep in range(step):
        for particles in all_particles:
            particles._push_position(0.5*dt)
            particles._push_momentum(dt)
            particles._push_position(0.5*dt)

            particles._calculate_chi()

            
            if particles.photon:
                photons = particles._radiate_photons(dt)
            if particles.pair:
                bw_pair = particles._create_pair(dt)
