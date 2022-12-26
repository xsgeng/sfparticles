from time import perf_counter_ns
from .fields import Fields
from .particles import Particles, c
from tqdm import tqdm

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

from . import _use_gpu
class Simulation(object):
    def __init__(self,
        *all_particles : Particles,
        dt: float,
        fields : Fields,
        print_every : int = 100,
        t0 = 0.0,
        photon_threshold = 2.0
    ) -> None:
        '''
        all_particles:
            specify all particles
        '''
        # TODO: check all inputs
        self.all_particles = all_particles
        self.particles_push = []
        self.particles_bw = []
        self.particles_rad = []
        self.dt = dt
        self.fields = fields
        self.print_every = print_every
        self.photon_threshold = photon_threshold
        
        self.t = t0
        self.step = 0

        self.progress_bar = None

        self.particles_dict = {p.name : p for p in self.all_particles}
        for p in all_particles:
            if p.bw:
                assert p.bw_electron in self.particles_dict, f"{p.bw_electron} not included in simulation."
                assert p.bw_positron in self.particles_dict, f"{p.bw_positron} not included in simulation."
                self.particles_bw.append(p)
            if p.radiating:
                assert p.photon in self.particles_dict
                self.particles_rad.append(p)
            if p.push:
                self.particles_push.append(p)


    def start(self, nstep, total=None):
        if total is None:
            total = nstep

        if self.progress_bar is None:
            if self.print_every:
                self.progress_bar = tqdm(total=total, unit='step', smoothing=1)


        if _use_gpu:
            for p in self.all_particles:
                p._to_gpu()

        tic = perf_counter_ns()
        for istep in range(self.step, self.step + nstep):
            # push particles
            for particles in self.particles_push:
                particles._eval_field(self.fields, self.t)

                # from t = (i-0.5)*dt to t = (i+0.5)*dt
                particles._push_momentum(self.dt)
                # from t = i*dt       to t = (i+0.5)*dt
                particles._push_position(0.5*self.dt)
                
            # QED
            for particles in self.particles_bw:
                particles._calculate_chi()
                particles._pair_event(self.dt)

            for particles in self.particles_rad:
                particles._calculate_chi()
                particles._photon_event(self.dt)
                particles._pick_hard_photon(self.photon_threshold)

            # create particles
            # seperated from events generation
            # since particles created in the current loop do NOT further create particle
            for particles in self.particles_rad:
                photon = self.particles_dict[particles.photon]
                particles._create_photon(photon)

            for particles in self.particles_bw:
                bw_electron = self.particles_dict[particles.bw_electron]
                bw_positron = self.particles_dict[particles.bw_positron]
                particles._create_pair(bw_electron, bw_positron)

            for particles in self.particles_push:
                # from t = (i+0.5)*dt to t = (i+1)*dt
                particles._push_position(0.5*self.dt)
            
            
            self.t += self.dt
            self.step += 1
            if self.print_every is None:
                continue
            elapsed = perf_counter_ns() - tic
            tic = perf_counter_ns()
            if self.step % self.print_every == 0:
                Ntotal = sum([particles.Npart for particles in self.all_particles])
                ns_per_particle = elapsed/Ntotal

                particle_num_str = ','.join([f"{particles.Npart} {particles.name}" for particles in self.all_particles])
                self.progress_bar.set_postfix_str(f"{particle_num_str}, {ns_per_particle:.0f}ns/particle")
                self.progress_bar.update(self.print_every)
                

        for particles in self.all_particles:
            if _use_gpu:
                particles._to_host()
            particles._prune()