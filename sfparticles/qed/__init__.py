from os import getenv

# set optical depth method from environmental variable
_use_optical_depth = False
if getenv("SFPARTICLES_OPTICAL_DEPTH") == "1":
    _use_optical_depth = True