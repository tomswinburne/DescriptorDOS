import numpy as np
import time,os

from mpi4py import MPI
from .YAMLParser import YAMLParser
from .LAMMPSWorker import LAMMPSWorker
from .TensorCompressor import TensorCompressor

class BasicAgent:
	"""
	BasicAgent class for initializing and managing a simulation environment.
	
	__init__(param_file='input.yaml', seed=None, setup=True, **kwargs): 	Initialize the BasicAgent with parameters from a file and optional keyword arguments.
	
	mpi_setup(world=None): Setup MPI environment.
	
	lammps_setup(comm=None, world=None, color=None): Setup LAMMPS instance. 
	
	close(): Close the LAMMPS worker.

	welcome_message(verbose=False): Print a welcome message.
	
	get_reference_data(): Return D_0 and Hessian information.
	
	run(target_keys, safe=True): Generate sample from LAMMPS worker.
	
	tensor_compression(D_data,old_res=None): Perform rank one compression.
	"""

	def __init__(self,param_file='input.yaml', seed=None, setup=True, **kwargs):
		"""
		Initialize the BasicAgent with parameters from a file and optional keyword arguments.

		Parameters:
		----------
		param_file (str): Path to the parameter file in YAML format. Default is 'input.yaml'.
		seed (int, optional): Seed for random number generation. If None, a seed is generated based on the current time.
		setup (bool): Flag to indicate whether to perform setup operations. Default is True.
		**kwargs: Additional keyword arguments to override parameters from the parameter file.

		Attributes:
		----------
		params (YAMLParser): Parsed parameters from the YAML file.

		control_parameters (dict): Dictionary of control parameters.

		input_parameters (dict): Dictionary of input parameters.
		
		seed (int): Seed for random number generation.
		
		directory (str): Directory path for dumping files.
		
		file_name (str): Full path to the pickle file for descriptor samples.
		
		default_calls (int): Default number of calls for some operation.
		
		stored_data (None): Placeholder for stored data.
		
		step (int): Current step in the simulation.
		
		Methods:
		--------
		mpi_setup(): Setup MPI environment.
		
		welcome_message(): Print a welcome message.
		
		file_setup(): Setup file-related configurations.
		
		lammps_setup(): Setup LAMMPS simulation environment.
		"""
	
		""" read parameter file """
		self.params = YAMLParser(yaml_file=param_file,**kwargs)

		for key,val in kwargs.items():
			self.params.input_params[key] = val
		if seed is None:
			self.seed = int(10000. * (time.time()-int(time.time())))
		else:
			self.seed = seed

		# Implemented control parameters with default values
		# each of these should be updated by apply_control_parameter
		self.control_parameters = {'Strain':0.0,'IsoSurface':1.0}
		

		# Input parameters with default values
		self.input_parameters = \
			self.params("input_parameters",{'IsoSurface':[0.05,2.0,2]})
		
		# Isosurface type
		self.isosurface = self.params("isosurface","Hessian")
		
		# Output directory
		self.directory = self.params("dump_path","./")
		self.hessian_path = self.params("hessian_path","./")

		# Output file
		self.file_name = self.params("dump_file","DescriptorSamples")
		self.file_name += f"-{self.isosurface}.pkl"
		self.file_name = os.path.join(self.directory,self.file_name)		

		# Default number of calls
		self.default_calls = self.params("default_calls",10)
		
		# Stored data
		self.stored_data = None
		
		# Current step
		self.step = 0

		if setup:
			self.mpi_setup()
			self.welcome_message()
			self.file_setup()
			self.lammps_setup()
			
	
	def mpi_setup(self,world=None):
		""" Setup MPI environment """
		self.rank = 0
		self.color = 0
		self.world = world
		self.comm = world
	
	def file_setup(self):
		""" Setup file-related configurations """
		os.system(f"mkdir -p {self.directory}")
		os.system(f"mkdir -p {self.hessian_path}")
	
	def lammps_setup(self,comm=None,world=None,color=None):
		"""
		Set up the LAMMPS worker with the provided or default communication parameters.
		
		Parameters:
		----------
		comm (optional): The MPI communicator. If None, defaults to self.comm.
		
		world (optional): The MPI world. If None, defaults to self.world.
		
		color (optional): The MPI color. If None, defaults to self.color.
		
		
		Attributes:
		----------
		lammps_worker (LAMMPSWorker): The worker instance for LAMMPS simulations.
		
		N (int): The initial number of particles.
		
		constants (dict): A copy of the constants from the LAMMPS worker.
		"""
		
		world = self.world if world is None else world
		comm = self.comm if comm is None else comm
		color = self.color if color is None else color
		self.lammps_worker = \
			LAMMPSWorker(comm,world,color,self.params,self.seed)
		self.N = self.lammps_worker.N_0
		self.strain = 0.0
		self.lammps_worker.reset()
		self.constants = self.lammps_worker.constants.copy()

	def welcome_message(self,verbose=False):
		if self.rank==0:
			print("""


\t\t************************************************************
\t\t\t\tD-DOS: Descriptor Density Of States
\t\t\t\tT. Swinburne, CNRS
\t\t\t\tMarinica Group, CEA
\t\t\t\t(C) MIT 2024
\t\t************************************************************


	""")
			if verbose:
				print("""	
				Parameters:
				""")
				for kk in self.params.input_params:
					print("\t",kk,self.params.input_params[kk])
	
	def get_reference_data(self):
		"""
			Return D_0 and Hessian information
		"""
		return self.lammps_worker.constants
	
	def change_volume(self,strain):
		"""
		Change the volume of the simulation box
		
		Parameters
		----------
		strain : float
			Strain value for the simulation box
		"""
		if np.abs(strain-self.strain)>1.e-6:
			self.lammps_worker.rescale_box(strain)
			self.strain = strain
		
	def calculate_hessian_modes(self):
		"""
		Calculate modes of Hessian matrix for this worker
		Returns:
		--------
		omega : numpy.ndarray
			The eigenvalues of the Hessian matrix.
		modes : numpy.ndarray
			The eigenvectors of the Hessian matrix.
		"""
		return self.lammps_worker.calculate_hessian_modes()
		
			
	def run(self,job,safe=False):
		"""
		Generate sample from lammps worker with specified target parameters

		Parameters
		----------
		target : dict
			If keys are in Workers's control_parameters, they will be applied
			
		safe : bool, optional
			calculate neighbor lists each call, by default True
		
		Returns
		-------
			res : dict
				Results from the LAMMPS worker, including sample_keys			
		"""
		sample_keys = job.copy()

		self.step += 1

		if not safe:
			run_safe = (self.step-1) % 100 == 0
		else:
			run_safe = True
		
		sample_keys['safe'] = run_safe
		
		# Isosurface sample takes IsoSurface as argument
		res = self.lammps_worker.isosurface_sample(sample_keys)
		return res
	
	
	def tensor_compression(self,D_data,old_D_data=None):
		"""
		Perform rank-one compression on the given data. 
		"""
		return TensorCompressor(D_data,self.N,
							CompressedData=old_D_data,
							bins = self.params("histogram_bins",100),
							bootstrap_ensemble_size = self.params("bootstrap_ensemble",10),
							threshold_percentile = self.params("histogram_percentile",1.0))
	
	def close(self):
		self.lammps_worker.close()
