from typing import Callable, List

from .fields import Fields
from .particles import Particles


def simulate(
    *all_particles : Particles,
    step: int,
    dt: float,
    field_function : Callable = None,
):
    field = Fields(field_function)
    t = 0.0
    for istep in range(step):
        for particles in all_particles:
            field._eval_field(particles, t)

            particles._push_position(0.5*dt)
            particles._push_momentum(dt)
            particles._push_position(0.5*dt)

            particles._calculate_chi()

            
            if particles.photon:
                photons = particles._radiate_photons(dt)
            if particles.pair:
                bw_pair = particles._create_pair(dt)
        t += dt*istep