import unittest
from scipy.constants import e, m_e, c, pi
import numpy as np
from sfparticles.particles import boris, vay

class TestPusher(unittest.TestCase):
    def test_boris(self):
        r = 1e-6
        ux0 = 100
        uz0 = 10
        u0 = np.sqrt(ux0**2 + uz0**2)
        v = ux0 / np.sqrt(1 + ux0**2) * c
        T = 2*pi*r/v

        bz = m_e*c/e * ux0 / r

        ux = np.asarray([ux0], dtype=float)
        uy = np.asarray([0], dtype=float)
        uz = np.asarray([uz0], dtype=float)
        inv_gamma = 1 / np.sqrt(1 + ux**2 + uy**2 + uz**2)

        Ex = np.asarray([0], dtype=float)
        Ey = np.asarray([0], dtype=float)
        Ez = np.asarray([0], dtype=float)

        Bx = np.asarray([0], dtype=float)
        By = np.asarray([0], dtype=float)
        Bz = np.asarray([bz], dtype=float)

        to_be_pruned = np.array([False])

        nt = 20
        for _ in range(nt):
            boris(ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, -e, 1, to_be_pruned, T/nt)
        
        self.assertLess(np.abs(np.sqrt(ux**2 + uy**2 + uz**2) - u0)/u0, 1E-10)

    def test_vay(self):
        r = 1e-6 #m
        ux0 = 1000
        uz0 = 10
        u0 = np.sqrt(ux0**2 + uz0**2)
        v = ux0 / np.sqrt(1 + ux0**2) * c
        T = 2*pi*r/v

        bz = m_e*c/e * ux0 / r

        ux = np.asarray([ux0], dtype=float)
        uy = np.asarray([0], dtype=float)
        uz = np.asarray([uz0], dtype=float)
        inv_gamma = 1 / np.sqrt(1 + ux**2 + uy**2 + uz**2)

        Ex = np.asarray([0], dtype=float)
        Ey = np.asarray([0], dtype=float)
        Ez = np.asarray([0], dtype=float)

        Bx = np.asarray([0], dtype=float)
        By = np.asarray([0], dtype=float)
        Bz = np.asarray([bz], dtype=float)

        to_be_pruned = np.array([False])

        nt = 20
        for _ in range(nt):
            vay(ux, uy, uz, inv_gamma, Ex, Ey, Ez, Bx, By, Bz, -e, 1, to_be_pruned, T/nt)

        self.assertLess(np.abs(np.sqrt(ux**2 + uy**2 + uz**2) - u0)/u0, 1E-10)