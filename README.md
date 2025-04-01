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

## Installation
`DescriptorDOS` uses <a href="https://docs.lammps.org/Python_head.html" target="_new">LAMMPS</a> and `mpi4py`,
in addition to `numpy` and `scipy`. If you can run
```python
from mpi4py import MPI
from lammps import lammps
lmp = lammps(comm=MPI.COMM_WORLD)
lmp.close()
```
Then you can install `DescriptorDOS`:
```bash
pip install DescriptorDOS
```
Scripts in `testing/unittests.py` to check everything is set up correctly.
Check out `examples/` for a simple application.
For HPC use we recommend installing `lammps` and `mpi4py` as described <a href="https://docs.lammps.org/Python_head.html" target="_new">here</a>

For **local** testing you can install LAMMPS via `conda-lammps` (one CPU/worker)
```bash
conda config --add channels conda-forge # add conda-forge channel
conda create -n ddos-env python=3.10 
conda activate ddos-env 
conda install mpi4py lammps
pip install DescriptorDOS
```
We emphasize `conda-lammps` will not give optimal performance on HPC!

## Algorithm Overview
The DescriptorDOS algorithm leverages MPI parallelism for efficient sampling and tensor-compression for low-rank storage. <br>
<img src="https://raw.githubusercontent.com/tomswinburne/DescriptorDOS/refs/heads/main/figures/algorithm.png" width=400></img>

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

