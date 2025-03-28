import numpy as np
import time,os
from mpi4py import MPI
from .YAMLParser import *
from .LAMMPSWorker import *
from .BasicAgent import *

class MPIAgent(BasicAgent):
	def __init__(self, world, param_file='input.yaml', seed=None, 
													setup=True,**kwargs):
		"""MPI wrapper of BasicAgent

		Parameters
		----------
		world : MPI.Intracomm
			MPI communicator
		param_file : os.PathLike[str], optional
			yaml parameter file, by default 'input.yaml'
		seed : None | int, optional
			random number seed, by default None
		"""
		BasicAgent.__init__(self,param_file=param_file,\
							seed=seed,setup=False, **kwargs)
		
		self.mpi_setup(world=world)
		self.welcome_message()
		
		if setup:
			self.finish_setup()

	def finish_setup(self):	
		""" 
			Finish setup of MPIAgent 
			Call after mpi_setup
		"""
		self.file_setup()
		self.lammps_setup()
		
	def close(self):
		""" 
			Close LAMMPS worker
		"""
		self.world.Barrier()
		if self.rank==0:
			self.lammps_worker.close()
		self.world.Barrier()
		

	def file_setup(self):
		""" 
			Setup file directory
		"""
		self.world.Barrier()
		if self.rank==0:
			super().file_setup()
		self.world.Barrier()
	
	def mpi_setup(self,world=None):
		""" 
			Setup MPI parameters
		"""
		self.world = world
		self.size = self.world.Get_size()
		self.rank = self.world.Get_rank()
		procs_per_worker = self.params("procs_per_worker")
		assert self.size % procs_per_worker == 0
		self.local_rank = self.rank % procs_per_worker
		self.color = self.rank // procs_per_worker
		self.nworkers = self.size // procs_per_worker
		self.comm = self.world.Split(self.color,0)
		self.roots = [i*procs_per_worker for i in range(self.nworkers)]
		self.ensemble_comm = self.world.Create(world.group.Incl(self.roots))
		self.seed += self.color
		self.default_calls = max(1,self.default_calls // self.nworkers)

	def lammps_setup(self):
		""" 
			Setup LAMMPS worker
		"""
		super().lammps_setup(comm=self.comm,world=self.world,color=self.color)
	

	def run(self,job,calls=None,time_limit=None):
		"""
		Parameters
		----------
		job : dict
			Keys to specify, must match self.data_keys
			Here, Strain and IsoSurface are specified
			
			IsoSurface is the isosurface value. Value of 1.0 approx equal to (3k_B/2) * 1000K in energy per degree of freedom
			
			Strain is the desired strain of the simulation box

		safe : bool, optional
			re-build neighbour lists or not, by default False

		Returns
		-------
		dict : dictionary of results			
		"""
		start = time.time()
		if not time_limit is None:
			check = lambda : (time.time()-start)>60.*time_limit
		else:
			check = lambda : False

		if calls is None:
			calls = self.default_calls if time_limit is None else 1000000

		indent = ""
		data = {}
		for call in range(calls):
			single_draw = super().run(job=job,safe=(call==0))
			if call==0:
				for k in single_draw:
					data[k] = [single_draw[k]]
			else:
				for k in single_draw:
					data[k] += [single_draw[k]]
			
			if check():
				break
			
			self.world.Barrier()			
			if self.rank==0:
				if call%(max(1,calls//20))==0:
					tcall = (call+1)*self.nworkers
					tcalls = calls*self.nworkers
					print(f"\r\t\t\t\tSampled {tcall}/{tcalls}", \
						end="", flush=True)
		self.world.Barrier()
		if self.rank==0:
			print("")
		
		return data

	def collate(self,data,old_res=None):
		"""
			collate arrays from generate() on rank 0
		"""

		self.world.Barrier()
		all_data = None
		if self.local_rank==0:
			all_data = {}
			for k in data:
				# TODO: This is a bit of a hack, should be more general
				if k=='D_tensor':
					data[k] = [np.array(data[k]).mean(0)] # formatting
				all_data[k] = self.ensemble_comm.gather(data[k],root=0)
		
		final_data = None
		new_calls = 0
		if self.rank==0:
			"""
				Currently, we simple put everything on rank 0 then compress
				Should obviously batch this ! 
			"""
			final_data = {}
			for k in all_data:
				final_data[k] = np.array(all_data[k][0])
				for f in all_data[k][1:]:
					f_data = np.array(f)
					final_data[k] = np.append(final_data[k],f_data,axis=0)
			del all_data
			new_calls = final_data['D'].shape[0]
		
		return final_data, new_calls