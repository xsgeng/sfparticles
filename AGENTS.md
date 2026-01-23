# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sfparticles** is a Python simulation toolkit for relativistic particle tracking in strong electromagnetic fields with QED effects.

- **Key features**: Numba acceleration, cascade simulation support, flexible field configuration, GPU acceleration (CUDA/cuPy)
- **Primary use case**: Scientific research in plasma physics, strong-field QED, laser-plasma interactions
- **Current version**: 0.4.12 (see `pyproject.toml`)
- **Python requirement**: >= 3.9

## Common Development Commands

### Installation and Setup

```bash
# Editable development installation
pip install -e .

# Build distribution packages
python -m build

# Install from PyPI
pip install sfparticles
```

### Testing

```bash
# Run all tests (uses unittest, not pytest)
python -m unittest discover -s tests -p "test_*.py"

# VS Code configuration supports unittest discovery (see .vscode/settings.json)
```

### Running Examples

```bash
# Basic trajectory simulation
python example/trajectory.py

# Cascade simulation with environment variables
SFPARTICLES_OPTICAL_DEPTH=1 python example/cascade.py

# GPU acceleration (requires CUDA/cuPy)
SFPARTICLES_USE_GPU=1 python example/cascade.py
```

### Environment Variables

- `SFPARTICLES_USE_GPU=1`: Enable GPU acceleration (requires CUDA/cuPy)
- `SFPARTICLES_OPTICAL_DEPTH=1`: Use optical depth method (default: rejection sampling)
- `NUMBA_NUM_THREADS`: Control CPU parallel thread count

## High-Level Architecture

### Core Components

1. **Simulation** (`simulation.py`): Orchestrates main loop, particle push, QED events, particle creation
2. **Particles** (`particles.py`): Manages particle data, includes `SpinParticles` for spin dynamics
3. **Fields** (`fields.py`): Electromagnetic field definitions with operator overloading for superposition
4. **CPU Backend** (`cpu.py`): Numba-jitted kernels for parallel CPU execution
5. **GPU Backend** (`gpu.py`): CUDA kernels for GPU acceleration
6. **QED Modules** (`qed/`): Optical depth and rejection sampling methods with pre-computed tables

### Simulation Flow

1. Field evaluation at current time
2. QED event generation (photon emission/pair production)
3. Particle creation from events
4. Particle push (2nd-order Boris scheme with optional radiation reaction)
5. Callback execution and progress reporting

### Particle Pusher Algorithms

- **Boris**: Standard relativistic pusher for Lorentz force
- **Boris-TBMT**: Includes Thomas-BMT spin precession
- **Landau-Lifshitz (LL)**: Continuous radiation reaction approximation
- **Quantum-corrected LL (CLL)**: LL with quantum suppression factor

**Radiation reaction types** (enum `RadiationReactionType`): `NONE`, `PHOTON`, `LL`, `CLL`. `PHOTON` enables discrete QED photon emission; `LL`/`CLL` provide continuous radiation reaction without photon emission.

### QED Integration

Two interchangeable methods selectable via environment variable:

1. **Optical-depth**: Particles carry optical depth τ; when τ < 0, event occurs
2. **Rejection-sampling**: Direct Monte Carlo sampling of QED differential rates

Both use pre-computed HDF5 tables (`*.h5` in `qed/`) for performance.

### Data Structures

- Particle attributes stored in contiguous NumPy/CuPy arrays
- 25% buffer overallocation for dynamic particle creation in cascades
- Pruning system for deleted particles (photons that pair-produce)

## Development Workflow

### GPU Setup Requirements

- Requires CUDA and cuPy installation: `conda install -c conda-forge cupy cudatoolkit=11.2`
- GPU is slower for small particle counts, better for large simulations

### Existing Claude Integration

- Skill `sfparticles-simulation` available in `.claude/skills/`
- Generates simulation scripts interactively or from templates
- Includes parameter validation and common pattern references
- See `.claude/skills/sfparticles-simulation/SKILL.md` for details

### Testing Patterns

- Uses unittest framework with VS Code configuration
- Tests cover particle initialization, pushers, QED optical depth, table operations
- No pytest configuration present

## Key Files Reference

### Source Files

- `sfparticles/__init__.py`: Public API (`Particles`, `SpinParticles`, `Simulation`, `Fields`, `RadiationReactionType`)
- `sfparticles/simulation.py`: Main simulation loop
- `sfparticles/particles.py`: Particle class and data management
- `sfparticles/fields.py`: Field definitions and composition
- `sfparticles/cpu.py`, `gpu.py`, `inline.py`: Performance kernels
- `sfparticles/qed/optical_depth.py`, `rejection_sampling.py`: QED event generators

### Configuration Files

- `pyproject.toml`: Modern Python packaging (PEP 517/518), dependencies, metadata
- `.vscode/settings.json`: unittest configuration for VS Code
- `MANIFEST.in`: Includes HDF5 data files in package distribution
- `.github/workflows/python-publish.yml`: GitHub Actions workflow for PyPI publishing on release

### Data Files

- `sfparticles/qed/optical_depth_tables.h5`: Pre-computed tables for optical depth method
- `sfparticles/qed/rejection_sampling_tables.h5`: Tables for rejection sampling method

## Notes and Conventions

- Python >= 3.9 required
- Uses absolute imports within package
- No `setup.py` - modern pyproject.toml-based build system
- HDF5 data files are included in package and essential for QED calculations
- Buffer management critical for cascade simulations with exponentially growing particle counts
- Current branch: `feat/sigmoid-sampling` (likely for upcoming sigmoid sampling feature)
- Recent commits focus on optical depth table optimization and interpolation improvements

## Additional Resources

- **README.md**: Installation, usage examples, GPU acceleration instructions
- **GitHub repository**: https://github.com/xsgeng/sfparticles
- **PyPI**: https://pypi.org/project/sfparticles/