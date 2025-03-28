import os,sys
import numpy as np
from .Constants import eV, Boltzmann, atomic_mass

from mpi4py import MPI

from .YAMLParser import YAMLParser
from .Displacers import Displacers

class BaseWorker:
	def __init__(self, comm, world, color, params, seed=None):
		"""
		Base class for LAMMPS worker with all direct LAMMPS calls removed.

		Parameters
		----------
		comm : MPI communicator for instance
		world : MPI communicator for simulation
		color : worker color
		params : YAMLParser object
			Dictionary of parameters, read from yaml file
		seed : int | None, optional
			Random number seed, by default None
		"""
		self.params = params
		self.directory = self.params("dump_path", "./")
		
		# some options
		self.verbose = self.params("verbose",False)
		self.compute_covar = self.params("compute_covar",False)
		

		if seed is not None:
			np.random.seed(seed)
		self.seed = np.random.randint(100, 2**8-1)
		

		self.comm = comm
		self.world = world
		self.color = color
		self.rank = 0 if world is None else world.Get_rank()
		self.size = 1 if world is None else world.Get_size()
		self.local_rank = 0 if comm is None else comm.Get_rank()
		self.local_size = 1 if comm is None else comm.Get_size()
		self.hessian_file = os.path.join(self.directory, f"dm_{self.rank}.dat")

		self.kB = Boltzmann / eV # eV/K


		# Placeholder for LAMMPS-specific attributes
		self.displacer = None
		self.constants = {}
		

		# Placeholder for state variables
		self.N_0 = None
		self.x = None
		self.x_0 = None
		self.mass = None
		self.potential = None
		self.N_D = None
		self.strain = float(0.0)
		self.zero_energy = None
		self.zero_descriptor = None
		self.constants = {}
		
	
	def set_constants(self):
		"""Placeholder for setting constants"""
		pass
	
	def update_constants(self, extra=None):
		"""Placeholder for updating constants"""
		if extra is not None:
			assert isinstance(extra, dict)
			self.constants.update(extra)
	
	def command(self, cmd_str):
		"""Placeholder for LAMMPS command execution"""
		pass

	def get_natoms(self):
		"""Placeholder for getting number of atoms"""
		return 0

	def get_positions(self):
		"""Placeholder for getting atom positions"""
		return np.array([])

	def get_mass(self):
		"""Placeholder for getting atom masses"""
		return np.array([])

	def get_potential_energy(self, offset=True):
		"""Placeholder for getting system energy"""
		return 0.0
	
	def get_kinetic_energy(self):
		"""Placeholder for getting system energy"""
		return 0.0
	
	def get_total_energy(self, offset=True):
		"""Placeholder for getting system energy"""
		return 0.0

	def get_descriptor(self):
		"""Placeholder for getting system descriptor"""
		return np.array([])

	def set_constants(self):
		"""Placeholder for setting constants"""
		pass

	def rescale_box(self, strain):
		"""Placeholder for rescaling simulation box"""
		pass
	
	def set_reference_volume(self, V=None):
		""" rescale box to match target volume """
		if V is None:
			return None
	def set_dummy_potential(self):
		"""Placeholder for setting dummy potential to accelerate Hessian-based sampling"""
		pass

	def calculate_hessian_modes(self):
		"""Placeholder for calculating Hessian matrix"""
		return np.array([[]])

	def displacement_from_x0(self, x=None):
		"""Placeholder for calculating displacement from initial positions"""
		return np.array([])
	
	def neighbor_shells(self, cutoff=10.):
		"""Placeholder for calculating neighbor shells"""
		return np.eye(1).astype(int)
	
	def run_nve(self, steps):
		"""Placeholder for running NVE simulation"""
		pass

	def set_nve(self):
		"""Placeholder for setting NVE """
		pass
	
	def target_energy(self, U):
		"""Placeholder for setting target energy"""
		pass

	def isosurface_sample(self, sample_keys):
		"""
			Sample isosurface, applying additional constraints

		Parameters
		----------
		sample_keys : dict
			Dictionary of parameters for the sample
			'IsoSurface', required
				isosurface value per atom
			'safe', optional
				if True, neighbor lists are recalculated
			'Strain', optional
				if provided, strain is set to this value
		
		Returns
		-------
		dict
			Dictionary containing sample data
		"""
		if 'Strain' in sample_keys:
			self.rescale_box(sample_keys['Strain'])
		
		if 'safe' in sample_keys:
			run_safe = sample_keys['safe']
		else:
			run_safe = True
		assert 'IsoSurface' in sample_keys, "'IsoSurface' key is required!"
		
		# pass worker to displacer in case of any changes
		self.displacer.prepare(self)
		
		# store the updated reference parameters
		self.update_constants(self.displacer.constants)
		
		sample = self.displacer(self,	isosurface_temperature=sample_keys['IsoSurface'],
										safe=run_safe)
		sample.update(sample_keys)
		
		sample['E'] = self.get_potential_energy(offset=True)
		sample['K'] = self.get_kinetic_energy()
		sample['D'] = self.get_descriptor()
		
		if self.compute_covar:
			sample['D_tensor'] = self.get_per_atom_descriptor_covariance_tensor()
		
		if self.verbose:
			print(f"\t\t\t\t{self.rank}: Sample: {sample['D'].shape[0]} descriptors, D1={sample['D'].flatten()[0]:.4g}, ,E={sample['E']:.4g} eV")
		return sample

	def close(self):
		"""Placeholder for closing LAMMPS instance"""
		pass