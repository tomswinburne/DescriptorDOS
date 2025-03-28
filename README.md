# DescriptorDOS
Score matching the descriptor density of states for model-agnostic free energy estimation

# Descriptor DOS

Code for paper 
*Score matching the descriptor density of states for model-agnostic free energy estimation*
[arXiv 2025](https://arxiv.org/abs/2502.18191)
<img src="https://raw.githubusercontent.com/tomswinburne/DescriptorDOS/refs/heads/main/cover-image.png" width=500></img>




- **[Installation](#conda-installation)**
- **[Example Usage](#usage)** 
- **[Helper Scripts](#helper-scripts)**
- **[Tips / Notes](#tips--notes)**

## Installing DescriptorDOS
DescriptorDOS requires `numpy`, `scipy` and two additional packages:
- `lammps` MD code, run via the python interface to evaluate descriptors and run NVE dynamics. 
- `mpi4py` for MPI-parallel sampling, with (optionally) multiple MPI ranks per `lammps` instance.

With a fresh python environment using e.g. `conda`:
```bash
conda create -n lammps_python_env python=3.10 
conda activate lammps_python_env # activate virtual env
pip install numpy scipy # install requirements 
```

Jan Janssen's `lammps` binary on `conda-force` has no GPU or internal MPI support, but allows easy testing
```bash 
conda config --add channels conda-forge # add conda-forge channel 
conda install mpi4py lammps # conda-lammps has no MPI: one core/worker!
```
For HPC use we recommend installing `lammps` and `mpi4py` as described <a href="https://docs.lammps.org/Python_head.html" target="_new">here</a>


Install `DescriptorDOS` with (`pip` coming soon!)
```bash 
git clone https://github.com/tomswinburne/DescriptorDOS.git
cd DescriptorDOS
python -m pip install -e . # will be on pypi asap....
```
You can run the tests in `testing/unittests.py` to check everything is set up correctly.

## Running Descriptor DOS
### Preparation
See `ExampleUsage/`.Some of the displacers require Hessian data. 
Parallel batching of these tasks will be done (simple python script),
 but for the moment
it is more efficient (i.e. maximum usage) to make these in serial before submission. 
For typical system sizes this requires max a few node minutes. 
i.e. 
```bash
python prepare_ddos.py # on login node
```
`ExampleUsage/prepare_ddos.py`:
```python
from DescriptorDOS import Manager 
DDOS_Manager = Manager(yaml_file="test-input-bcc.yaml")
DDOS_Manager.test_run()
```
We can then run the DDOS sampling in parallel:
```bash
mpirun -np ${NPROCS} python run_ddos.py
```
`run_ddos.py`:
```python
from mpi4py import MPI
from DescriptorDOS import Manager 
DDOS_Manager = Manager( comm=MPI.COMM_WORLD,
                        yaml_file="test-input-bcc.yaml")
DDOS_Manager.run()
```

`test-input-bcc.yaml`:
```yaml
# Output directory
dump_path: output/ddos-bcc-test

# Data file - will append isosurface type and save as pickle file
dump_file: histograms-bcc

# Dictionary of adjustable parameters with range in low,high,steps
# IsoSurface is the isosurface value, 1.0 corresponds to T~=1000K
# See the isosurface_sample function of DescriptorDOS/Analyzer.py
input_parameters:
  IsoSurface: [0.05, 2.0, 2]  # exp alpha bounds
  Strain: [0.0, 0.04, 2]  # volumetric strain bounds


# Isosurface per atom is U_0 exp(alpha)
# U_0 set such that T~=1000K for exp(alpha)=1
iso_value_bound : [0.05,2.0,2]

# Type of isosurface: "Hessian" "Kinetic" or "Basic"
isosurface : "Kinetic"

# TOTAL force calls for each value of alpha
default_calls : 10

# Histogram bins per degree of freedom
histogram_bins : 100

# CPU cores per worker
procs_per_worker : 1 # Optimal value is 1 unless memory limited

# pickle file for Hessian data- will be read if present in dump_path
hessian_data : "Hessian-Data-All-Strains.pkl" # in dump_path

# MUST correspond to compute in LAMMPS script below!
global_descriptor_compute_name : aveD

# Must provide a compute global_descriptor_compute_name 
# which returns a global average descriptor of size descriptor_size 
lammps_input_script: in.lammps
```


The `lammps` input file must provide a compute matching `global_descriptor_compute_name`
which returns a global average descriptor of size `descriptor_size`

`in.lammps`:
```bash
  clear
  atom_style atomic
  atom_modify map array sort 0 0.0
  units           metal
  boundary        p p p
  
  read_data Configurations/Fe_BCC_333.dat
  mass * 183.84

  pair_style zero 4.7
  #pair_coeff * * Potentials/X_Fe.snapcoeff Potentials/D.snapparam X

  thermo          1
  thermo_modify norm no

  neighbor 2.0 bin
  neigh_modify once no every 1 delay 0 check yes

  compute D all sna/atom 4.7 0.99363 8 0.5 1
  compute aveD all reduce ave c_D[*]

  reset_timestep	0
  run           	0
```

### Helper scripts
- Simple tests for DDOS pacakge in `testing/unittests.py`
- Two generation scripts, using `ase` and `matscipy` installed via `pip` in addition to `lammps`:
- Equilibrium lattice structures for some potential in `prepare/Make_Configurations.ipynb`
- Hypercube UQ parameter uncertainties in `ReferenceSystems/ASEConstraints_HyperCube_DesMat.ipynb`

### Tips / Notes
- **Results are unchanged** for a given value of $r_{\rm cut}/a_{\rm lat}$ with Bispectrum descriptors

(c) MIT 2024 TD Swinburne thomas.swinburne@cnrs.fr

