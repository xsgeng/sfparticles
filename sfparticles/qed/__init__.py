import os
_use_optical_depth = False
if "SFPARTICLES_OPTICAL_DEPTH" in os.environ:
    if os.environ["SFPARTICLES_OPTICAL_DEPTH"] == "1":
        _use_optical_depth = True

if _use_optical_depth:
    print("Using optical depth method.")

from .optical_depth import update_tau_e, update_tau_gamma
from .rejection_sampling import pair_from_rejection_sampling, photon_from_rejection_sampling