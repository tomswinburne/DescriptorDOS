<img src="https://raw.githubusercontent.com/tomswinburne/DescriptorDOS/refs/heads/main/figures/cover-image.png" width=500></img>

DescriptorDOS is a scheme to return phase free energies as smooth functions of MLIP parameters,
for uncertainty quantification in forward propagation and inverse fine-tuning in back-propagation.

Details are provided in our paper:<br>
*Score matching the descriptor density of states for model-agnostic free energy estimation*<br>
Preprint:[arXiv 2025](https://arxiv.org/abs/2502.18191)<br>

DescriptorDOS is designed for the wide variety of MLIPs that are linear in some set of descriptor features.  
As we discuss in the paper, whilst this clearly includes ACE/SNAP/qSNAP/POD/MILADY "as-is", multiple
applications to non-linear architectures e.g. fine-tuning MACE are possible and will be available in the near future. 
**Further applications / demonstrations of `DescriptorDOS` are coming- watch this space!**

**The present implementation requires a `LAMMPS` descriptor calculation- see below**

## Algorithm Overview
The DescriptorDOS algorithm leverages MPI parallelism for efficient sampling and tensor-compression for low-rank storage. <br>
<img src="https://raw.githubusercontent.com/tomswinburne/DescriptorDOS/refs/heads/main/figures/algorithm.png" width=400></img>

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
See `examples/`. Whilst sampling is fully parallel, to generate samples with `Hessian` displacer, 
we require the reference system's Hessian. 

Parallel batching of these tasks is simple but current implementation simply runs in 
serial for 5-10 minutes before batch submission (see `examples/run.py`):
```bash
python run.py # will store Hessian data and only run a few samples
```

We can then run the DDOS sampling in parallel:
```bash
mpirun -np ${NPROCS} python run.py
```

Whilst we provide many options in `run.py`, they can all be stored in a `yaml` file,
so the python script can be as succinct as 
```python
from mpi4py import MPI
from DescriptorDOS import Manager 
DDOS_Manager = Manager( comm=MPI.COMM_WORLD,
                        yaml_file="configuration.yaml")
DDOS_Manager.run()
```

## Analysing data
We provide a sample notebooks in `analysis/` of our "world-first" result, inverse fine-tuning of a phase transition temperature. 

More examples will be added in the coming months. 

(c) MIT 2024 TD Swinburne thomas.swinburne@cnrs.fr

