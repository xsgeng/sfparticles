import os
import sys

os.environ['SFPARTICLES_OPTICAL_DEPTH'] = '1'
del sys.modules['sfparticles.qed']
del sys.modules['sfparticles.particles']
    
from test_qed import TestPairNumber, TestPhotonNumber
from sfparticles.particles import _use_optical_depth


class TestPairNumberOD(TestPairNumber):
    def setUp(self) -> None:
        self.assertTrue(_use_optical_depth)
        super().setUp()
        
        
class TestPhotonNumberOD(TestPhotonNumber):
    def setUp(self) -> None:
        self.assertTrue(_use_optical_depth)
        super().setUp()