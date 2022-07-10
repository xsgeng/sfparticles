import os

import numpy as np
from scipy.constants import alpha, pi, m_e, hbar, c
from scipy.special import airy
from scipy.integrate import quad
import h5py
import multiprocessing

with h5py.File(os.path.join(os.path.dirname(__file__), 'tables.h5'), 'r') as f:
    dset = f['photon_prob_rate_total']
    _prob_rate_table = dset[()]
    _chi_N = dset.attrs['chi_N']
    _chi_range = dset.attrs['log_chi_range']
    _chi_delta = dset.attrs['log_chi_delta']


def Ai(z):
    return airy(z)[0]

def Aip(z):
    return airy(z)[1]


def int_Ai(z):
    return quad(Ai, z, np.inf)[0]

def prob_rate_for_delta(chi_e):
    factor = -alpha*m_e*c**2/hbar
    def prob_(delta):
        z = (delta/(1-delta)/chi_e)**(2/3)
        g = 1 + delta**2/2/(1-delta)
        return factor*(int_Ai(z) + g*2/z * Aip(z))

    return prob_

def integral_over_delta(chi_e):
    P = prob_rate_for_delta(chi_e)
    prob_rate_total, _ = quad(P, 0, 1)
    return prob_rate_total


def prob_rate_for_chi_delta(chi_e, delta):
    factor = -alpha*m_e*c**2/hbar
    z = (delta/(1-delta)/chi_e)**(2/3)
    g = 1 + delta**2/2/(1-delta)
    return factor*(int_Ai(z) + g*2/z * Aip(z))


def photon_prob_rate_total(chi_N=256, log_chi_min=-3, log_chi_max=2):
    with multiprocessing.Pool() as pool:
        data = pool.map(integral_over_delta, np.logspace(log_chi_min, log_chi_max, chi_N))
    return np.array(data)


def table_gen(table_path, chi_N=256, log_chi_min=-3, log_chi_max=2):
    with h5py.File(os.path.join(table_path, 'tables.h5'), 'w') as h5f:
        print("integrating photon emission probability rate over delta")
        dset = h5f.create_dataset('photon_prob_rate_total', data=photon_prob_rate_total(chi_N, log_chi_min, log_chi_max))
        dset.attrs['chi_N'] = chi_N
        dset.attrs['log_chi_range'] = (log_chi_min, log_chi_max)
        dset.attrs['log_chi_delta'] = (log_chi_max - log_chi_min) / (chi_N - 1)

if __name__ == '__main__':
    table_gen(os.path.dirname(__file__))