
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
table_path = os.path.join(os.path.dirname(__file__), 'optical_depth_tables.h5')
if os.path.exists(table_path) and __name__ == "sfparticles.qed.optical_depth_tables":
    with h5py.File(table_path, 'r') as f:
        # 1d
        _photon_prob_rate_total_table = f['photon_prob_rate_total'][()]
        _pair_prob_rate_total_table = f['pair_prob_rate_total'][()]
        # 2d
        _integral_photon_prob_along_delta = f['integral_photon_prob_along_delta'][()]
        _integral_pair_prob_along_delta = f['integral_pair_prob_along_delta'][()]

        _chi_N = f.attrs['chi_N']
        _log_chi_range = f.attrs['log_chi_range']
        _log_chi_delta = f.attrs['log_chi_delta']
        _delta_N = f.attrs['delta_N']
        _log_delta_range = f.attrs['log_delta_range']
        _log_delta_delta = f.attrs['log_delta_delta']

    del f

@njit
def _get_chi_idx(chi):
    log_chi = np.log10(chi)
    if log_chi < _log_chi_range[0]:
        return -1
    if log_chi > _log_chi_range[1]:
        idx = _chi_N - 1
    if _log_chi_range[0] <= log_chi <= _log_chi_range[1]:
        idx = (log_chi - _log_chi_range[0]) / _log_chi_delta

    return idx

@njit
def _linear_interp1d(chi, table1d):
    idx = math.floor(_get_chi_idx(chi))
    if idx == -1:
        return 0.0
    log_chi_left = _log_chi_range[0] + idx*_log_chi_delta
    # linear interp
    log_chi = np.log10(chi)
    k = (table1d[idx+1] - table1d[idx]) / _log_chi_delta
    prob_rate = table1d[idx] + k * (log_chi-log_chi_left)
    return prob_rate


@njit
def _bisect_interp(chi, table2d):
    low, high = 0, _delta_N-1
    chi_idx = _get_chi_idx(chi)
    chi_idx_low = math.floor(chi_idx)
    chi_idx_high = chi_idx_low + 1

    chi_low = 10**(_log_chi_range[0] + chi_idx_low * _log_chi_delta)
    chi_high = 10**(_log_chi_range[0] + chi_idx_high * _log_chi_delta)
    
    k = (table2d[chi_idx_high,  0] - table2d[chi_idx_low,  0]) / (chi_high - chi_low)
    ymin = table2d[chi_idx_low,  0] + k * (chi - chi_low)

    k = (table2d[chi_idx_high, -1] - table2d[chi_idx_low, -1]) / (chi_high - chi_low)
    ymax = table2d[chi_idx_low, -1] + k * (chi - chi_low)

    # lower than ymin are ignored, NOTE: too large ymin may cause
    # higher number
    r = np.random.rand() * (ymax-ymin) + ymin
    while low <= high:
        mid = int((low + high)/2)
        k = (table2d[chi_idx_high, mid] - table2d[chi_idx_low, mid]) / (chi_high - chi_low)
        mid_delta = table2d[chi_idx_low, mid] + k * (chi - chi_low)

        if mid_delta < r:
            low = mid + 1
        elif mid_delta > r:
            high = mid - 1
    
    # interp
    delta_idx = high # high = low - 1, the left index

    k = (table2d[chi_idx_high, delta_idx] - table2d[chi_idx_low, delta_idx]) / (chi_high - chi_low)
    y1 = table2d[chi_idx_low, delta_idx] + k * (chi - chi_low)

    k = (table2d[chi_idx_high, delta_idx+1] - table2d[chi_idx_low, delta_idx+1]) / (chi_high - chi_low)
    y2 = table2d[chi_idx_low, delta_idx+1] + k * (chi - chi_low)

    delta_left = 10**(_log_delta_range[0] + delta_idx*_log_delta_delta)
    delta_right = 10**(_log_delta_range[0] + (delta_idx+1)*_log_delta_delta)
    k = (delta_right - delta_left) / (y2 - y1)
    delta = delta_left + k * (r - y1)

    return delta
    

@njit
def integ_photon_prob_rate_from_table(chi_e):
    return _linear_interp1d(chi_e, _photon_prob_rate_total_table)


@njit
def integ_pair_prob_rate_from_table(chi_gamma):
    return _linear_interp1d(chi_gamma, _pair_prob_rate_total_table)


@njit
def photon_delta_from_chi_delta_table(chi_e):
    return _bisect_interp(chi_e, _integral_photon_prob_along_delta)

@njit
def pair_delta_from_chi_delta_table(chi_gamma):
    return _bisect_interp(chi_gamma, _integral_pair_prob_along_delta)

'''
table generations
'''
def Ai(z):
    return airy(z)[0]

def Aip(z):
    return airy(z)[1]


def int_Ai(z):
    return quad(Ai, z, np.inf)[0]

def gen_photon_prob_rate_for_delta(chi_e):
    factor = -alpha*m_e*c**2/hbar
    def prob_(delta):
        if delta == 1.0: 
            return 0.0
        chi_gamma = delta * chi_e
        chi_ep = chi_e - chi_gamma
        z = (chi_gamma/chi_e/chi_ep)**(2/3)
        return factor*(int_Ai(z) + (2.0/z + chi_gamma*np.sqrt(z)) * Aip(z))

    return prob_

def gen_pair_prob_rate_for_delta(chi_gamma):
    factor = alpha*m_e*c**2/hbar
    def prob_(delta):
        if delta == 1.0: 
            return 0.0
        chi_e = delta * chi_gamma
        chi_ep = chi_gamma - chi_e
        z = (chi_gamma/chi_e/chi_ep)**(2/3)
        return factor*(int_Ai(z) + (2.0/z - chi_gamma*np.sqrt(z)) * Aip(z))

    return prob_

def integral_photon_prob_over_delta(chi_e):
    P = gen_photon_prob_rate_for_delta(chi_e)
    prob_rate_total, _ = quad(P, 0, 1)
    return prob_rate_total

def integral_pair_prob_over_delta(chi_gamma):
    P = gen_pair_prob_rate_for_delta(chi_gamma)
    prob_rate_total, _ = quad(P, 0, 1)
    return prob_rate_total

def integral_photon_prob_along_delta(chi_e, delta_N, log_delta_min):
    P = gen_photon_prob_rate_for_delta(chi_e)
    delta = np.logspace(log_delta_min, 0, delta_N)
    integ = np.zeros(delta_N)
    # 积分从delta_min开始
    integ[0] = quad(P, 0, delta[0])[0]
    for i in range(1, delta_N):
        integ[i] = integ[i-1] + P(delta[i]) * (delta[i] - delta[i-1])
    return integ

def integral_pair_prob_along_delta(chi_gamma, delta_N, log_delta_min):
    P = gen_pair_prob_rate_for_delta(chi_gamma)
    delta = np.logspace(log_delta_min, 0, delta_N)
    integ = np.zeros(delta_N)
    # 积分从delta_min开始
    integ[0] = quad(P, 0, delta[0])[0]
    for i in range(1, delta_N):
        integ[i] = integ[i-1] + P(delta[i]) * (delta[i] - delta[i-1])
    return integ

def photon_prob_rate_total(chi_N=256, log_chi_min=-3, log_chi_max=2):
    with multiprocessing.Pool() as pool:
        data = pool.map(integral_photon_prob_over_delta, np.logspace(log_chi_min, log_chi_max, chi_N))
    return np.array(data)

def pair_prob_rate_total(chi_N=256, log_chi_min=-3, log_chi_max=2):
    with multiprocessing.Pool() as pool:
        data = pool.map(integral_pair_prob_over_delta, np.logspace(log_chi_min, log_chi_max, chi_N))
    return np.array(data)


def table_gen(
    table_path, 
    chi_N=256, log_chi_min=-3.0, log_chi_max=2.0, 
    delta_N=1024, log_delta_min=-6,
):
    with h5py.File(os.path.join(table_path, 'optical_depth_tables.h5'), 'w') as h5f:

        print("计算不同chi_e的总辐射概率")
        h5f.create_dataset('photon_prob_rate_total', data=photon_prob_rate_total(chi_N, log_chi_min, log_chi_max))

        print("计算不同chi_gamma的总电子对概率")
        h5f.create_dataset('pair_prob_rate_total', data=pair_prob_rate_total(chi_N, log_chi_min, log_chi_max))

        chi = np.logspace(log_chi_min, log_chi_max, chi_N)
        print("计算不同chi_e辐射概率的积分")
        with multiprocessing.Pool() as pool:
            integ = pool.starmap(integral_photon_prob_along_delta, zip(chi, [delta_N]*chi_N, [log_delta_min]*chi_N))
        h5f.create_dataset('integral_photon_prob_along_delta', data=integ)

        print("计算不同chi_gamma电子对概率的积分")
        with multiprocessing.Pool() as pool:
            integ = pool.starmap(integral_pair_prob_along_delta, zip(chi, [delta_N]*chi_N, [log_delta_min]*chi_N))
        h5f.create_dataset('integral_pair_prob_along_delta', data=integ)

        h5f.attrs['chi_N'] = chi_N
        h5f.attrs['log_chi_range'] = (log_chi_min, log_chi_max)
        h5f.attrs['log_chi_delta'] = (log_chi_max - log_chi_min) / (chi_N - 1)
        h5f.attrs['delta_N'] = delta_N
        h5f.attrs['log_delta_range'] = (log_delta_min, 0)
        h5f.attrs['log_delta_delta'] = (0 - log_delta_min) / (delta_N - 1)


if __name__ == '__main__':
    table_gen(os.path.dirname(__file__))