import unittest

import sfparticles
from scipy.constants import e, m_e, c, pi
import numpy as np

from sfparticles import Particles
from itertools import product

class TestParticlesInit(unittest.TestCase):
    def test_particles_init(self):
        p = Particles('e', q=-1, m=1, N=2)
        self.assertEqual(p.q, -e)
        self.assertEqual(p.m, m_e)
        self.assertEqual(p.N_buffered, 2)

        for prop, i in product(['', 'u', 'E', 'B'], ['x', 'y', 'z']):
            attr = prop + i
            with self.subTest(attr):
                self.assertTrue((getattr(p, attr) == np.zeros(2)).all())
        

    def test_spin_init(self):
        p = Particles('e', q=-1, m=1, N=2, has_spin=True)
        self.assertTrue(p.has_spin)

        self.assertTrue(hasattr(p, 'sx'))
        self.assertTrue(hasattr(p, 'sy'))
        self.assertTrue(hasattr(p, 'sz'))

        self.assertTrue((p.sx == np.zeros(2)).all)
        self.assertTrue((p.sy == np.zeros(2)).all)
        self.assertTrue((p.sz == np.ones(2)).all)

    
    def test_prop_init_list(self):
        p = Particles('e', q=-1, m=1, N=3, props=[[1, 2, 3]]*6)
        target = np.array([1, 2, 3], dtype=np.float64)
        
        self.assertEqual(p.N_buffered, 3)

        for prop, i in product(['', 'u'], ['x', 'y', 'z']):
            attr = prop + i
            with self.subTest(f"init {attr}"):
                self.assertTrue((getattr(p, attr) == target).all(), msg=f"{attr} = {getattr(p, attr)}")

    def test_prop_init_scalar(self):

        p = Particles('e', q=-1, m=1, N=1, props=[1]*6)
        target = np.array([1], dtype=np.float64)

        self.assertEqual(p.N_buffered, 1)

        for prop, i in product(['', 'u'], ['x', 'y', 'z']):
            attr = prop + i
            with self.subTest(f"init {attr}"):
                self.assertTrue((getattr(p, attr) == target).all(), msg=f"{attr} = {getattr(p, attr)}")


class TestParticleResize(unittest.TestCase):
    def test_append(self):
        N = 5
        N_new1 = 10
        N_new2 = 30
        p = Particles('e', 1, 1, 5, has_spin=True)

        p._append(([1]*N_new1)*6, N_new1)
        self.assertEqual(p.N_buffered, N + N_new1)
        self.assertEqual(p.buffer_size, 2*N + N_new1)
        self.assertEqual(p.x.shape[0], p.buffer_size)

        p._append(([2]*N_new2)*6, N_new2)
        self.assertEqual(p.N_buffered, N + N_new1 + N_new2)
        self.assertEqual(p.buffer_size, (2*N + N_new1) + N_new2 + (N + N_new1))


        attrs = [prop + i for prop, i in product(['', 'u', 'E', 'B'], ['x', 'y', 'z'])] + \
            ['inv_gamma', 'optical_depth']
        for attr in attrs:
            with self.subTest(attr):
                self.assertEqual(len(getattr(p, attr)), p.buffer_size)

        attrs = [prop + i for prop, i in product(['', 'u'], ['x', 'y', 'z'])] 
        for attr in attrs:
            with self.subTest(attr):
                self.assertEqual(getattr(p, attr)[N:N+N_new1].tolist(), [1]*N_new1)
                self.assertEqual(getattr(p, attr)[N+N_new1:N+N_new1+N_new2].tolist(), [2]*N_new2)


        

