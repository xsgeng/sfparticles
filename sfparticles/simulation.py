from time import perf_counter_ns
from .fields import Fields
from .particles import Particles, c



class Simulation(object):
    def __init__(self,
        *all_particles : Particles,
        dt: float,
        fields : Fields,
        print_every : int = 100,
        t0 = 0.0,
    ) -> None:
        self.all_particles = all_particles
        self.dt = dt
        self.fields = fields
        self.print_every = print_every
        self.t = t0
        self.step = 0


    def start(self, nstep):
        self.tic = perf_counter_ns()
        for istep in range(self.step, self.step + nstep):
            # push particles
            for particles in self.all_particles:
                particles._eval_field(self.fields, self.t)

                particles._push_position(0.5*self.dt)
                particles._push_momentum(self.dt)
                particles._push_position(0.5*self.dt)

            # QED
            for particles in self.all_particles:
                particles._calculate_chi()

                if hasattr(particles, 'pair'):
                    particles._pair_event(self.dt)
                if hasattr(particles, 'photon'):
                    particles._photon_event(self.dt)
                    
            # create particles
            # seperated from events generation
            # since particles created in the current loop do NOT further create particle
            for particles in self.all_particles:
                if hasattr(particles, 'photon_delta'):
                    particles._create_photon()
                if hasattr(particles, 'pair_delta'):
                    particles._create_pair()
            
            self.t += self.dt
            if (istep+1) % self.print_every == 0 :
                elapsed = perf_counter_ns() - self.tic
                self.tic = perf_counter_ns()

                Ntotal = sum([particles.Npart for particles in self.all_particles])
                print(
                    f'step: {istep+1}, ct: {c*self.t/1e-6:.2f} um, ',
                    ', '.join([f"{particles.Npart} {particles.name}" for particles in self.all_particles]),
                    f', {elapsed/self.print_every/Ntotal:.2f} ns/particle'
                )

        for particles in self.all_particles:
            particles._prune()