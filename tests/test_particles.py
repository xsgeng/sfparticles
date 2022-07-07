import unittest

import sfparticles
from scipy.constants import e, m_e, c, pi
import numpy as np

from sfparticles import particles
from itertools import product

class TestParticlesInit(unittest.TestCase):
    def test_particles_init(self):
        p = sfparticles.Particles('e', q=-1, m=1, N=2)
        self.assertEqual(p.q, -e)
        self.assertEqual(p.m, m_e)
        self.assertEqual(p.N, 2)

        for prop, i in product(['', 'u', 'E', 'B'], ['x', 'y', 'z']):
            attr = prop + i
            with self.subTest(attr):
                self.assertTrue((getattr(p, attr) == np.zeros(2)).all())
        

    def test_spin_init(self):
        p = sfparticles.Particles('e', q=-1, m=1, N=2, has_spin=True)
        self.assertTrue(p.has_spin)

        self.assertTrue(hasattr(p, 'sx'))
        self.assertTrue(hasattr(p, 'sy'))
        self.assertTrue(hasattr(p, 'sz'))

        self.assertTrue((p.sx == np.zeros(2)).all)
        self.assertTrue((p.sy == np.zeros(2)).all)
        self.assertTrue((p.sz == np.ones(2)).all)

    
    def test_prop_init_list(self):
        p = sfparticles.Particles('e', q=-1, m=1, N=3, props=[[1, 2, 3]]*6)
        target = np.array([1, 2, 3], dtype=np.float64)
        
        self.assertEqual(p.N, 3)

        for prop, i in product(['', 'u'], ['x', 'y', 'z']):
            attr = prop + i
            with self.subTest(f"init {attr}"):
                self.assertTrue((getattr(p, attr) == target).all(), msg=f"{attr} = {getattr(p, attr)}")

    def test_prop_init_scalar(self):

        p = sfparticles.Particles('e', q=-1, m=1, N=1, props=[1]*6)
        target = np.array([1], dtype=np.float64)

        self.assertEqual(p.N, 1)

        for prop, i in product(['', 'u'], ['x', 'y', 'z']):
            attr = prop + i
            with self.subTest(f"init {attr}"):
                self.assertTrue((getattr(p, attr) == target).all(), msg=f"{attr} = {getattr(p, attr)}")


