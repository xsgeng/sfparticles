from os import environ
_use_gpu = False
if "SFPARTICLES_USE_GPU" in environ:
    if environ["SFPARTICLES_USE_GPU"] == "1":
        print("using GPU")
        _use_gpu = True