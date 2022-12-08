import os

import numpy as np
from numba import njit, cfunc
import math
from scipy.constants import alpha, pi, m_e, hbar, c
from scipy.special import airy
from scipy.integrate import quad
import h5py
import multiprocessing


# built-in tables
table_path = os.path.join(os.path.dirname(__file__), 'tables.h5')
if os.path.exists(table_path) and __name__ == "sfparticles.qed.tables":
    with h5py.File(table_path, 'r') as f:
        dset = f['photon_prob_rate_total']

        dset = f['int_Ai']
        _int_Ai_table = dset[()]
        dset = f['Aip']
        _Aip_table = dset[()]

        _z_N = dset.attrs['z_N']
        _z_range = dset.attrs['z_range']
        _z_delta = dset.attrs['z_delta']

    del f, dset


'''
Rejection_sampling
'''
@njit
def photon_prob_rate_from_table(chi_e, delta):
    chi_gamma = delta * chi_e
    chi_ep = chi_e - chi_gamma
    z = (chi_gamma/chi_e/chi_ep)**(2/3)
    factor = -alpha*m_e*c**2/hbar

    if z < _z_range[0]:
        raise ValueError('z < 0')
    if z > _z_range[1]:
        idx = _z_N - 1
    if _z_range[0] <= z <= _z_range[1]:
        idx = math.floor((z - _z_range[0]) / _z_delta)
    # linear interp
    z_left = _z_range[0] + idx * _z_delta

    k = (_int_Ai_table[idx+1] - _int_Ai_table[idx]) / _z_delta
    int_Ai_ = _int_Ai_table[idx] + k * (z - z_left)

    k = (_Aip_table[idx+1] - _Aip_table[idx]) / _z_delta
    Aip_ = _Aip_table[idx] + k * (z - z_left)
    
    return factor*(int_Ai_ + (2.0/z + chi_gamma*np.sqrt(z)) * Aip_)


@njit
def pair_prob_rate_from_table(chi_gamma, delta):
    chi_e = delta * chi_gamma
    chi_ep = chi_gamma - chi_e
    z = (chi_gamma/chi_e/chi_ep)**(2/3)
    factor = alpha*m_e*c**2/hbar

    if z < _z_range[0]:
        raise ValueError('z < 0')
    if z > _z_range[1]:
        return 0.0
    if _z_range[0] <= z <= _z_range[1]:
        idx = math.floor((z - _z_range[0]) / _z_delta)
    # linear interp
    z_left = _z_range[0] + idx * _z_delta

    k = (_int_Ai_table[idx+1] - _int_Ai_table[idx]) / _z_delta
    int_Ai_ = _int_Ai_table[idx] + k * (z - z_left)

    k = (_Aip_table[idx+1] - _Aip_table[idx]) / _z_delta
    Aip_ = _Aip_table[idx] + k * (z - z_left)
    
    # - for pair
    return factor*(int_Ai_ + (2.0/z - chi_gamma*np.sqrt(z)) * Aip_)

def Ai(z):
    return airy(z)[0]

def Aip(z):
    return airy(z)[1]


def int_Ai(z):
    return quad(Ai, z, np.inf)[0]


def table_gen(
    table_path, 
    chi_N=256, log_chi_min=-3.0, log_chi_max=2.0, 
    z_N=1024, z_min=0.0, z_max=100.0
):
    with h5py.File(os.path.join(table_path, 'tables.h5'), 'w') as h5f:
        print("Integrating Ai")
        z = np.linspace(z_min, z_max, z_N)

        with multiprocessing.Pool() as pool:
            _int_Ai = pool.map(int_Ai, z)
        dset = h5f.create_dataset('int_Ai', data=_int_Ai)
        dset.attrs['z_N'] = z_N
        dset.attrs['z_range'] = (z_min, z_max)
        dset.attrs['z_delta'] = (z_max - z_min) / (z_N - 1)

        print("Ai'")
        _Aip = Aip(z)
        dset = h5f.create_dataset('Aip', data=_Aip)
        dset.attrs['z_N'] = z_N
        dset.attrs['z_range'] = (z_min, z_max)
        dset.attrs['z_delta'] = (z_max - z_min) / (z_N - 1)


if __name__ == '__main__':
    table_gen(os.path.dirname(__file__))