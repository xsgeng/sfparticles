import os

import numpy as np
from numba import njit
import math
from scipy.constants import alpha, pi, m_e, hbar, c
from scipy.special import airy
from scipy.integrate import quad
import h5py
import multiprocessing


# built-in tables
table_path = os.path.join(os.path.dirname(__file__), 'tables.h5')
if os.path.exists(table_path):
    with h5py.File(table_path, 'r') as f:
        dset = f['photon_prob_rate_total']
        _photon_prob_rate_total_table = dset[()]
        _chi_N = dset.attrs['chi_N']
        _chi_range = dset.attrs['log_chi_range']
        _chi_delta = dset.attrs['log_chi_delta']

        dset = f['int_Ai']
        _int_Ai_table = dset[()]
        dset = f['Aip']
        _Aip_table = dset[()]

        _z_N = dset.attrs['z_N']
        _z_range = dset.attrs['z_range']
        _z_delta = dset.attrs['z_delta']

    del f, dset


'''
Optical depth
'''
@njit
def integ_prob_rate_from_table(chi_e):
    log_chi_e = np.log10(chi_e)
    if log_chi_e < _chi_range[0]:
        return 0.0
    if log_chi_e > _chi_range[1]:
        idx = _chi_N - 1
    if _chi_range[0] <= log_chi_e <= _chi_range[1]:
        idx = math.floor((log_chi_e - _chi_range[0]) / _chi_delta)

    log_chi_e_left = _chi_range[0] + idx*_chi_delta
    # linear interp
    k = (_photon_prob_rate_total_table[idx+1] - _photon_prob_rate_total_table[idx]) / _chi_delta
    prob_rate = _photon_prob_rate_total_table[idx] + k * (log_chi_e-log_chi_e_left)

    return prob_rate


# TODO
@njit
def delta_from_chi_delta_table(chi_e):
    # dummy
    return np.random.rand()


'''
Rejection_sampling
'''
@njit
def prob_rate_from_table(chi_e, delta):
    factor = -alpha*m_e*c**2/hbar
    z = (delta/(1-delta)/chi_e)**(2/3)
    g = 1 + delta**2/2/(1-delta)

    if z < _z_range[0]:
        raise ValueError('z < 0')
    if z > _z_range[1]:
        idx = _z_N - 1
    if _z_range[0] <= z <= _z_range[1]:
        idx = math.floor((z - _z_range[0]) / _z_delta)
    # linear interp
    z_left = _z_range[0] + idx * _z_delta

    k = (_int_Ai_table[idx+1] - _int_Ai_table[idx]) / _chi_delta
    int_Ai_ = _int_Ai_table[idx] + k * (z - z_left)

    k = (_Aip_table[idx+1] - _Aip_table[idx]) / _chi_delta
    Aip_ = _Aip_table[idx] + k * (z - z_left)
    
    return factor*(int_Ai_ + g*2/z * Aip_)

def Ai(z):
    return airy(z)[0]

def Aip(z):
    return airy(z)[1]


def int_Ai(z):
    return quad(Ai, z, np.inf)[0]

def gen_prob_rate_for_delta(chi_e):
    factor = -alpha*m_e*c**2/hbar
    def prob_(delta):
        z = (delta/(1-delta)/chi_e)**(2/3)
        g = 1 + delta**2/2/(1-delta)
        return factor*(int_Ai(z) + g*2/z * Aip(z))

    return prob_

def integral_over_delta(chi_e):
    P = gen_prob_rate_for_delta(chi_e)
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
        _Aip = airy(z)[1]
        dset = h5f.create_dataset('Aip', data=_Aip)
        dset.attrs['z_N'] = z_N
        dset.attrs['z_range'] = (z_min, z_max)
        dset.attrs['z_delta'] = (z_max - z_min) / (z_N - 1)

        print("计算不同chi_e的总辐射概率")
        dset = h5f.create_dataset('photon_prob_rate_total', data=photon_prob_rate_total(chi_N, log_chi_min, log_chi_max))
        dset.attrs['chi_N'] = chi_N
        dset.attrs['log_chi_range'] = (log_chi_min, log_chi_max)
        dset.attrs['log_chi_delta'] = (log_chi_max - log_chi_min) / (chi_N - 1)

        # TODO: 2D chi-delta

if __name__ == '__main__':
    table_gen(os.path.dirname(__file__))