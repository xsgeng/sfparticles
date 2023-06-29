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
    

def boris_tbmt_inline( ux, uy, uz, sx, sy, sz, Ex, Ey, Ez, Bx, By, Bz, q, ae, dt ) :

    efactor = q*dt/(2*m_e*c)
    bfactor = q*dt/(2*m_e)

    # E field
    ux_minus = ux + efactor * Ex
    uy_minus = uy + efactor * Ey
    uz_minus = uz + efactor * Ez
    # B field
    # spin first
    u_dot_B = ux_minus*Bx + uy_minus*By + uz_minus*Bz

    gamma_minus = sqrt( 1.0 + ux_minus**2 + uy_minus**2 + uz_minus**2 )
    inv_gamma_minus = 1. / gamma_minus
    Bx_eff = (ae + inv_gamma_minus) * Bx - ae * gamma_minus / (gamma_minus + 1) * u_dot_B * ux_minus - (ae + 1 / (gamma_minus + 1)) * (uy_minus * Ez - uz_minus * Ey)
    By_eff = (ae + inv_gamma_minus) * By - ae * gamma_minus / (gamma_minus + 1) * u_dot_B * uy_minus - (ae + 1 / (gamma_minus + 1)) * (uz_minus * Ex - ux_minus * Ez)
    Bz_eff = (ae + inv_gamma_minus) * Bz - ae * gamma_minus / (gamma_minus + 1) * u_dot_B * uz_minus - (ae + 1 / (gamma_minus + 1)) * (ux_minus * Ey - uy_minus * Ex)

    Tx = bfactor * Bx_eff
    Ty = bfactor * By_eff
    Tz = bfactor * Bz_eff

    T2 = Tx * Tx + Ty * Ty + Tz * Tz

    Sx = 2 * Tx / (1. + T2)
    Sy = 2 * Ty / (1. + T2)
    Sz = 2 * Tz / (1. + T2)

    # s' = si + si x T
    sx_plus = sx + sy * Tz - sz * Ty
    sy_plus = sy + sz * Tx - sx * Tz
    sz_plus = sz + sx * Ty - sy * Tx

    # sf = si + s' x S
    sx_new = sx + sy_plus * Sz - sz_plus * Sy
    sy_new = sy + sz_plus * Sx - sx_plus * Sz
    sz_new = sz + sx_plus * Sy - sy_plus * Sx

    # then momentum
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

    ux_new = ux_plus + efactor * Ex
    uy_new = uy_plus + efactor * Ey
    uz_new = uz_plus + efactor * Ez
    inv_gamma_ = 1 / sqrt(1 + ux_new**2 + uy_new**2 + uz_new**2)
    return ux_new, uy_new, uz_new, inv_gamma_, sx_new, sy_new, sz_new
    


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