from .particles import Particles, RadiationReactionType
from .simulation import Simulation
from .fields import Fields

__all__ = ["Particles", "Simulation", "Fields", "RadiationReactionType"]

from os import environ
_use_gpu = False
if "SFPARTICLES_USE_GPU" in environ:
    if environ["SFPARTICLES_USE_GPU"] == "1":
        _use_gpu = True