from time import perf_counter
import unittest
from scipy.special import airy
from scipy.integrate import quad_vec, quad
from scipy.constants import alpha, m_e, c, hbar, e
from numba import njit, prange, jit
import numpy as np

from sfparticles.particles import Particles
from sfparticles.fields import static_field
from sfparticles.qed.optical_depth_tables import integral_photon_prob_over_delta, integral_pair_prob_over_delta

class TestPhotonNumber(unittest.TestCase):
    def setUp(self) -> None:
        self.gamma = 10000
        ux = np.sqrt(self.gamma**2 - 1)
        self.N = 1_000_000
        
        self.ele = Particles(
            'ele', -1, 1, N=self.N,
            props=(
                np.zeros(self.N), np.zeros(self.N), np.zeros(self.N), 
                np.full(self.N, ux), np.zeros(self.N), np.zeros(self.N), 
            ),
        )
        
        self.Bfield_from_chi = lambda chi_e : static_field(Bz=chi_e * m_e**2*c**2/e/hbar / ux)
        
        return super().setUp()
    
    def test_photon_creation(self):
        dt = 1e-17
        
        Bfield = self.Bfield_from_chi(1)
        
        ele = self.ele
        pho = Particles('pho', 0, 0)
        ele.set_photon(pho)
        
        ele._eval_field(Bfield, 0)
        ele._calculate_chi()
        
        ele._photon_event(dt)
        ele._create_photon(pho)
        ele._photon_event(dt)
        ele._create_photon(pho)
        self.assertEqual(pho.N_buffered, pho.Npart)
        
    def test_hard_photon(self):
        dt = 1e-17
        Bfield = self.Bfield_from_chi(1)
        
        ele = self.ele
        pho = Particles('pho', 0, 0)
        ele.set_photon(pho)
        
        ele._eval_field(Bfield, 0)
        ele._calculate_chi()
        
        ele._photon_event(dt)
        ele._pick_hard_photon(2.0)
        ele._create_photon(pho)
        self.assertTrue((pho.gamma > 2.0).all())
        
    def test_photon_number(self):
        interval = 1e-17
        tau = interval / self.gamma
        for chi_e in [0.1, 0.5, 1.0, 2.0, 5.0]:
            with self.subTest(chi_e=chi_e):
                Bfield = self.Bfield_from_chi(chi_e)
                
                ele = self.ele
                pho = Particles('pho', 0, 0)
                ele.set_photon(pho)
                
                ele._eval_field(Bfield, 0)
                ele._calculate_chi()
                
                
                ele._photon_event(interval)
                ele._create_photon(pho)
                n_photon = pho.Npart
                
                prob_rate_total = integral_photon_prob_over_delta(chi_e) * tau
                n_photon_expected = prob_rate_total * self.N
                sigma = np.sqrt(n_photon_expected * (1 - prob_rate_total))
                tor = 3*sigma

                self.assertLess(abs(n_photon-n_photon_expected), tor, f'n_photon={n_photon}, n_photon_expected={n_photon_expected}')


class TestPairNumber(unittest.TestCase):
    def setUp(self) -> None:
        self.gamma = 10000
        ux = np.sqrt(self.gamma**2 - 1)
        self.N = 1_000_000
        
        self.pho = Particles(
            'pho', 0, 0, N=self.N,
            props=(
                np.zeros(self.N), np.zeros(self.N), np.zeros(self.N), 
                np.full(self.N, ux), np.zeros(self.N), np.zeros(self.N), 
            ),
        )
        
        self.Bfield_from_chi = lambda chi_e : static_field(Bz=chi_e * m_e**2*c**2/e/hbar / ux)
        
        return super().setUp()
    
    def test_pair_creation(self):
        dt = 1e-17
        
        Bfield = self.Bfield_from_chi(1)
        
        pho = self.pho
        ele = Particles('ele', -1, 1)
        pos = Particles('pos', 1, 1)
        pho.set_pair(ele, pos)
        
        pho._eval_field(Bfield, 0)
        pho._calculate_chi()
        
        pho._pair_event(dt)
        pho._create_pair(ele, pos)
        pho._pair_event(dt)
        pho._create_pair(ele, pos)

        self.assertEqual(pos.N_buffered, pos.Npart)
        self.assertEqual(ele.N_buffered, ele.Npart)
        self.assertEqual(pos.N_buffered, ele.Npart)
        self.assertEqual(self.N - pho.Npart, pos.Npart)
    
    def test_momentum(self):
        dt = 1e-17
        
        Bfield = self.Bfield_from_chi(1)
        
        pho = self.pho
        ele = Particles('ele', -1, 1)
        pos = Particles('pos', 1, 1)
        pho.set_pair(ele, pos)
        
        pho._eval_field(Bfield, 0)
        pho._calculate_chi()
        
        pho._pair_event(dt)
        ux_pho = pho.ux[pho.event]
        uy_pho = pho.uy[pho.event]
        uz_pho = pho.uz[pho.event]
        
        pho._create_pair(ele, pos)
        
        self.assertTrue((abs(ele.ux + pos.ux - ux_pho) < 1E-10).all())
        self.assertTrue((abs(ele.uy + pos.uy - uy_pho) < 1E-10).all())
        self.assertTrue((abs(ele.uz + pos.uz - uz_pho) < 1E-10).all())
    
    def test_pair_number(self):
        interval = 1e-17
        tau = interval / self.gamma
        for chi_gamma in [1.0, 2.0, 5.0]:
            with self.subTest(chi_gamma=chi_gamma):
                Bfield = self.Bfield_from_chi(chi_gamma)
                
                pho = self.pho
                ele = Particles('ele', -1, 1)
                pos = Particles('pos', 1, 1)
                pho.set_pair(ele, pos)
                
                pho._eval_field(Bfield, 0)
                pho._calculate_chi()

                pho._pair_event(interval)
                
                pho._create_pair(ele, pos)
                n_pair = pos.Npart
                
                prob_rate_total = integral_pair_prob_over_delta(chi_gamma) * tau
                n_pair_expected = prob_rate_total * self.N
                sigma = np.sqrt(n_pair_expected * (1 - prob_rate_total))
                tor = 3*sigma

                self.assertLess(abs(n_pair-n_pair_expected), tor, f'n_pair={n_pair}, n_pair_expected={n_pair_expected}')