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
        photon_threshold = 2.0
    ) -> None:
        '''
        all_particles:
            specify all particles
        '''
        # TODO: check all inputs
        self.all_particles = all_particles
        self.dt = dt
        self.fields = fields
        self.print_every = print_every
        self.photon_threshold = photon_threshold
        
        self.t = t0
        self.step = 0


    def start(self, nstep):
        particles_dict = {p.name : p for p in self.all_particles}
        self.tic = perf_counter_ns()
        for istep in range(self.step, self.step + nstep):
            # push particles
            for particles in self.all_particles:
                particles._eval_field(self.fields, self.t)

                # from t = (i-0.5)*dt to t = (i+0.5)*dt
                particles._push_momentum(self.dt)
                # from t = i*dt       to t = (i+0.5)*dt
                particles._push_position(0.5*self.dt)
                
            # QED
            for particles in self.all_particles:
                particles._calculate_chi()

                if hasattr(particles, 'bw_electron'):
                    particles._pair_event(self.dt)
                if hasattr(particles, 'photon'):
                    particles._photon_event(self.dt)
                    particles._pick_hard_photon(self.photon_threshold)

            # create particles
            # seperated from events generation
            # since particles created in the current loop do NOT further create particle
            for particles in self.all_particles:
                if hasattr(particles, 'photon_delta'):
                    photon = particles_dict[particles.photon]
                    particles._create_photon(photon)
                if hasattr(particles, 'pair_delta'):
                    bw_electron = particles_dict[particles.bw_electron]
                    bw_positron = particles_dict[particles.bw_positron]
                    particles._create_pair(bw_electron, bw_positron)

            for particles in self.all_particles:
                # from t = (i+0.5)*dt to t = (i+1)*dt
                particles._push_position(0.5*self.dt)
            
            
            self.t += self.dt
            self.step += 1
            if self.print_every is None:
                continue
            if (istep+1) % self.print_every == 0 :
                elapsed = perf_counter_ns() - self.tic
                self.tic = perf_counter_ns()

                Ntotal = sum([particles.Npart for particles in self.all_particles])
                print(
                    f'step: {istep+1}\tct: {c*self.t/1e-6:.2f} um\t',
                    '\t'.join([f"{particles.Npart} {particles.name}" for particles in self.all_particles]),
                    f'\t{elapsed/self.print_every/Ntotal:.2f} ns/particle',
                    f'\t{elapsed/1E9:.2f} s'
                )

        for particles in self.all_particles:
            particles._prune()