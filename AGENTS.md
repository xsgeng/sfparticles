# sfparticles Agent Notes

## Setup and Verification
- Install the package and runtime dependencies with `python -m pip install -e .` (Python >=3.9; dependencies are declared only in `pyproject.toml`).
- Tests use `unittest`, not a configured pytest/tooling stack. Run one module with `python -m unittest tests.test_pusher`; discover the suite with `python -m unittest discover -s tests`.
- `tests/test_qed.py` allocates up to 10,000,000 particles and checks random event statistics; `test_qed_optical_depth.py` repeats those tests. Their statistical assertions can intermittently fail, so rerun a failure before treating it as a regression. Do not use full discovery as a quick check.
- No lint, formatter, typecheck, CI workflow, or task runner is configured.

## Architecture and Runtime Modes
- Public API is exported from `sfparticles/__init__.py`: `Particles`, `SpinParticles`, `Simulation`, `Fields`, and `RadiationReactionType`.
- `Simulation.start()` is the orchestration point: evaluate fields, perform QED events, create emitted particles/pairs, then apply the second-order position/momentum push. Particle collections connected through `set_photon()` or `set_pair()` must also be passed to `Simulation`.
- `Fields` compiles supplied scalar field functions, which must accept `(x, y, z, t)` and return `(Ex, Ey, Ez, Bx, By, Bz)`. It evaluates each particle at `t + t_offset`.
- CPU kernels are in `cpu.py`/`inline.py`; GPU kernels are separate in `gpu.py`. Keep CPU and GPU implementations aligned when changing shared particle behavior.
- `SFPARTICLES_USE_GPU=1` is read during import. It requires both Numba CUDA and CuPy, and dispatches particle/field work to GPU code. Set it before importing `sfparticles`.
- `SFPARTICLES_OPTICAL_DEPTH=1` is also read during import and switches QED event generation from rejection sampling to optical-depth tables. Set it before importing; tests that switch it reload modules explicitly.
- The bundled QED `.h5` tables in `sfparticles/qed/` are runtime inputs and are explicitly included in source distributions by `MANIFEST.in`; do not omit them when changing packaging or table loading.
