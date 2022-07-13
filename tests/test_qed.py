from time import perf_counter
import unittest
from scipy.special import airy
from scipy.integrate import quad_vec, quad
from scipy.constants import alpha, m_e, c, hbar, e
from numba import njit, prange, jit
import numpy as np

from sfparticles.particles import Particles
from sfparticles.fields import static_field


def Ai(z):
    return airy(z)[0]

def Aip(z):
    return airy(z)[1]


def int_Ai(z):
    return quad(Ai, z, np.inf)[0]

def photon_prob_rate_delta(chi_e):
    factor = -alpha*m_e*c**2/hbar
    def prob_(delta):
        z = (delta/(1-delta)/chi_e)**(2/3)
        g = 1 + delta**2/2/(1-delta)
        return factor*(int_Ai(z) + g*2/z * Aip(z))

    return prob_

def pair_prob_rate_delta(chi_gamma):
    factor = alpha*m_e*c**2/hbar
    def prob_(delta):
        chi_e = delta * chi_gamma
        chi_ep = chi_gamma - chi_e
        z = (chi_gamma/chi_e/chi_ep)**(2/3)
        return factor*(int_Ai(z) + (2.0/z - chi_gamma*np.sqrt(z)) * Aip(z))

    return prob_


class TestPhotonNumber(unittest.TestCase):
    def test_photon_number(self):
        tor = 0.1 # 10%
        gamma = 10000
        ux = np.sqrt(gamma**2 - 1)
        Bfield_from_chi = lambda chi_e : static_field(Bz=chi_e * m_e**2*c**2/e/hbar / ux)

        N = 1_000_000
        p = Particles(
            'e', -1, 1, N=N,
            props=(
                np.zeros(N), np.zeros(N), np.zeros(N), 
                np.full(N, ux), np.zeros(N), np.zeros(N), 
            ),
        )

        interval = 1e-17
        tau = interval / gamma
        for chi_e in [0.1, 0.5, 1.0, 2.0, 5.0]:
            with self.subTest(chi_e=chi_e):
                Bfield = Bfield_from_chi(chi_e)
                p._eval_field(Bfield, 0)
                p._calculate_chi()

                event, _ = p._photon_event(interval)
                n_photon = event.sum()/N
                
                P = photon_prob_rate_delta(chi_e)
                prob_rate_total, err = quad(P, 0, 1)
                n_photon_expected = prob_rate_total * tau

                self.assertLess(abs(n_photon-n_photon_expected)/n_photon_expected, tor, f'n_photon={n_photon}, n_photon_expected={n_photon_expected}')
    
    def test_pair_number(self):
        tor = 0.1 # 10%
        gamma = 10000
        ux = gamma
        Bfield_from_chi = lambda chi_gamma : static_field(Bz=chi_gamma * m_e**2*c**2/e/hbar / ux)

        N = 1_000_000
        p = Particles(
            'pho', 0, 0, N=N,
            props=(
                np.zeros(N), np.zeros(N), np.zeros(N), 
                np.full(N, ux), np.zeros(N), np.zeros(N), 
            ),
        )

        interval = 1e-16
        tau = interval / gamma
        for chi_gamma in [1.0, 2.0, 5.0]:
            with self.subTest(chi_gamma=chi_gamma):
                Bfield = Bfield_from_chi(chi_gamma)
                p._eval_field(Bfield, 0)
                p._calculate_chi()

                event, _ = p._pair_event(interval)
                n_pair = event.sum()/N
                
                P = pair_prob_rate_delta(chi_gamma)
                prob_rate_total, err = quad(P, 0, 1)
                n_pair_expected = prob_rate_total * tau


                self.assertLess(abs(n_pair-n_pair_expected)/n_pair_expected, tor, f'n_pair={n_pair}, n_pair_expected={n_pair_expected}')