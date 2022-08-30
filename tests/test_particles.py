import unittest

import sfparticles
from scipy.constants import e, m_e, c, pi
import numpy as np

from sfparticles import Particles
from itertools import product

from sfparticles.particles import RadiationReactionType

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
        
        with self.assertRaises(AssertionError):
            Particles('pho', q=1, m=0, N=2)

        with self.assertRaises(AssertionError):
            Particles('ele', q=-1, m=1, RR='LL')

    def test_set_photon(self):
        ele = Particles('ele', q=-1, m=1, RR=RadiationReactionType.LL)
        pho = Particles('pho', q=0, m=0, N=2)
        
        with self.assertRaises(AssertionError):
            ele.set_photon(pho)

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
    def test_extend(self):
        N = 5
        N_new1 = 10
        N_new2 = 30
        e = Particles('e', -1, 1, N, has_spin=True)
        pho = Particles('pho', 0, 0)
        e.set_photon(pho)

        e._extend(N_new1)
        self.assertEqual(e.N_buffered, N)
        self.assertEqual(e.buffer_size, int(N/4) + N + N_new1)
        self.assertEqual(e.x.shape[0], e.buffer_size)

        e._extend(N_new2)
        self.assertEqual(e.N_buffered, N)
        self.assertEqual(e.buffer_size, (int(N/4) + N + N_new1) + int(N/4) + N_new2 )


        attrs = [prop + i for prop, i in product(['', 'u', 's', 'E', 'B'], ['x', 'y', 'z'])] + \
            ['inv_gamma', 'optical_depth', '_to_be_pruned', 'event', 'photon_delta', 'event_index']
        for attr in attrs:
            with self.subTest(attr):
                self.assertEqual(len(getattr(e, attr)), e.buffer_size)


        

