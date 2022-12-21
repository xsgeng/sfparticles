import os
os.environ['SFPARTICLES_OPTICAL_DEPTH'] = '1'

from test_qed import TestPairNumber, TestPhotonNumber 

class TestPairNumberOD(TestPairNumber):
    pass
class TestPhotonNumberOD(TestPhotonNumber):
    pass