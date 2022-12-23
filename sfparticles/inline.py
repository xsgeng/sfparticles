from .particles import m_e, c, pi, epsilon_0, hbar, e
from math import sqrt

def boris_inline( ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, q, dt ) :

    efactor = q*dt/(2*m_e*c)
    bfactor = q*dt/(2*m_e)

    # E field
    ux_minus = ux + efactor * Ex
    uy_minus = uy + efactor * Ey
    uz_minus = uz + efactor * Ez
    # B field
    inv_gamma_minus = 1 / sqrt(1 + ux_minus**2 + uy_minus**2 + uz_minus**2)
    Tx = bfactor * Bx * inv_gamma_minus
    Ty = bfactor * By * inv_gamma_minus
    Tz = bfactor * Bz * inv_gamma_minus
    
    ux_prime = ux_minus + uy_minus * Tz - uz_minus * Ty
    uy_prime = uy_minus + uz_minus * Tx - ux_minus * Tz
    uz_prime = uz_minus + ux_minus * Ty - uy_minus * Tx

    Tfactor = 2 / (1 + Tx**2 + Ty**2 + Tz**2)
    Sx = Tfactor * Tx
    Sy = Tfactor * Ty
    Sz = Tfactor * Tz

    ux_plus = ux_minus + uy_prime * Sz - uz_prime * Sy
    uy_plus = uy_minus + uz_prime * Sx - ux_prime * Sz
    uz_plus = uz_minus + ux_prime * Sy - uy_prime * Sx

    ux_ = ux_plus + efactor * Ex
    uy_ = uy_plus + efactor * Ey
    uz_ = uz_plus + efactor * Ez
    inv_gamma_ = 1 / sqrt(1 + ux_**2 + uy_**2 + uz_**2)
    return ux_, uy_, uz_, inv_gamma_
    


def LL_push_inline( ux, uy, uz, inv_gamma, chi_e, dt ) :
    factor = -2/3 / (4*pi*epsilon_0) * e**2 * m_e * c / hbar**2 * dt
        
    ux_ = ux + factor * chi_e**2 * ux*inv_gamma
    uy_ = uy + factor * chi_e**2 * uy*inv_gamma
    uz_ = uz + factor * chi_e**2 * uz*inv_gamma
    inv_gamma_ = 1 / sqrt(1 + ux_**2 + uy_**2 + uz_**2)
    return ux_, uy_, uz_, inv_gamma_
    

def calculate_chi_inline(Ex, Ey, Ez, Bx, By, Bz, ux, uy, uz, inv_gamma):
    factor = e*hbar / (m_e**2 * c**3)
    gamma = 1.0 / inv_gamma
    return factor * sqrt(
        (gamma*Ex + (uy*Bz - uz*By)*c)**2 +
        (gamma*Ey + (uz*Bx - ux*Bz)*c)**2 +
        (gamma*Ez + (ux*By - uy*Bx)*c)**2 -
        (ux*Ex + uy*Ey + uz*Ez)**2
    )




            



