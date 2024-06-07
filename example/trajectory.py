from sfparticles import Particles, Simulation
from sfparticles.fields import simple_laser_pulse
from sfparticles.particles import RadiationReactionType

import matplotlib.pyplot as plt

import numpy as np
from scipy.constants import pi, m_e, e, c


um = 1e-6
fs = 1e-15

l0 = 0.8*um
omega0 = 2*pi*c / l0
k0 = omega0 / c

a0 = 150
w0 = 5*um
ctau = 5*um

a0_norm = e / (m_e * c * omega0)

N = int(1)
Tsim = 5*ctau/c
dt = 1*um / c / 100
step = int(Tsim / dt)
gen = np.random.RandomState()


gamma0 = 100

def init(N):
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    ux = np.full(N, np.sqrt(gamma0**2-1))
    uy = np.zeros(N)
    uz = np.zeros(N)
    return (x, y, z, ux, uy, uz)

def simulate(N=1, RR=RadiationReactionType.LL):
    laser = simple_laser_pulse(a0, w0, ctau, direction=-1, x0=3*ctau)

    electrons = Particles('electron', q=-1, m=1, N=N, props=init(N), RR=RR)
    photons = Particles('photon', 0, 0)

    if RR == RadiationReactionType.PHOTON:
        electrons.set_photon(photons)

    sim = Simulation(electrons, photons, dt=dt, fields=laser, print_every=100)

    trajectory_size = (step, N)
    t = np.arange(step) * dt
    x = np.zeros(trajectory_size)
    y = np.zeros(trajectory_size)
    z = np.zeros(trajectory_size)
    ux = np.zeros(trajectory_size)
    uy = np.zeros(trajectory_size)
    uz = np.zeros(trajectory_size)
    
    def store_trajectory(i):
        x[i] = electrons.x
        y[i] = electrons.y
        z[i] = electrons.z
        ux[i] = electrons.ux
        uy[i] = electrons.uy
        uz[i] = electrons.uz

    sim.start(step, call_back=store_trajectory)

    return t, x, y, z, ux, uy, uz


def main():
    fig, axes = plt.subplots(
        2, 1,
        tight_layout = True,
        figsize=(5, 5),
    )

    ax = axes[0]
    for RR in [
        RadiationReactionType.NONE, 
        RadiationReactionType.LL, 
    ]:
        t, x, y, z, ux, uy, uz = simulate(1, RR)
        ax.plot(
            x/um, y/um,
            label = f'RR = {RR.name}',
        )

    ax.set_xlim(0, 15)
    ax.legend()

    ax = axes[1]
    t, x, y, z, ux, uy, uz = simulate(20, RadiationReactionType.PHOTON)
    ax.plot(
        x/um, y/um,
        lw = 1,
        color = 'k',
    )
    ax.set_xlim(0, 15)
    fig.savefig('trajectory.png', dpi=300)

if __name__ == "__main__":
    main()