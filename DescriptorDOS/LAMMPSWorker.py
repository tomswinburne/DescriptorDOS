
from .BaseWorker import BaseWorker
from .Displacers import Displacers
import numpy as np
import importlib.util
import os
from lammps import lammps, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR

# The LAMMPSWorker class inherits from BaseWorker 
class LAMMPSWorker(BaseWorker):
	"""
	LAMMPSWorker class inherits from BaseWorker and implements LAMMPS-specific functionality.
	"""
	def __init__(self, comm, world, color, params, seed=None):
		"""
		Initialize the LAMMPSWorker class.
		
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
		super().__init__(comm, world, color, params, seed)
		
		# LAMMPS-specific initialization here
		# Initialize LAMMPS-specific attributes
		self.vector_compute_name = params("atom_descriptor_compute_name", "D")
		self.compute_name = params("global_descriptor_compute_name", "aveD")
		self.start_lammps()

		# if provided, set reference lattice constant
		self.set_reference_volume(self.params('V_target', None))

		
		# Initialize basic state variables
		self.N_0 = self.get_natoms()
		self.x = self.get_positions()
		self.C = self.get_cell()
		self.x_0 = self.x.copy()
		self.shell_mat,self.shell_num = self.neighbor_shells()
		self.mass = self.get_mass()
		
		# Initialize potential data
		coeff_file = self.params('potential_coefficient_file', None)
		if coeff_file is not None:
			coeff_file = coeff_file.strip().split()
			self.potential = \
				np.loadtxt(coeff_file[0], skiprows=int(coeff_file[1]))
		else:
			self.potential = None
			self.N_D = None
		
		# Initialize strain and energy
		self.strain = float(0.0)
		self.zero_energy = self.get_potential_energy(offset=False)
		
		# Initialize descriptor
		try:
			self.zero_descriptor = self.get_descriptor()
			if self.compute_covar:
				self.zero_descriptor_tensor = self.get_per_atom_descriptor_covariance_tensor()
		except:
			print(Exception)
			raise ValueError(f"Could not extract descriptor! Check LAMMPS input script for {self.compute_name} compute.")
		self.N_D = self.zero_descriptor.size
		
		# Set constants and initialize displacer
		self.set_constants()
		isosurface = params("isosurface", "Hessian")
		assert isosurface in Displacers.keys()
		self.displacer = Displacers[isosurface](self)
		self.update_constants(self.displacer.constants.copy())
		
		# For NVE ensemble
		self.nve_set = False
	
	# LAMMPS-specific methods
	def start_lammps(self):
		"""Initialize LAMMPS instance"""
		lammps_spec = importlib.util.find_spec("lammps")
		if lammps_spec is None:
			raise ValueError("LAMMPS not found!")
		
		# Set LAMMPS output options
		if self.params("verbose",False):
			instance = self.rank // self.local_size
			cmdargs=["-screen","none","-log",f"log_{instance}.lammps"]
		else:
			cmdargs=["-screen","none","-log","none"]
		
		# Initialize LAMMPS instance
		lammps_path = lammps_spec.origin
		if 'conda' in lammps_path.lower():
			if self.rank==0:
				print("\t\t\tLAMMPS compiled with conda, 1 core/worker\n")
			try:
				self.lmp = lammps(cmdargs=cmdargs, comm=None)
			except:
				raise ValueError("Error initializing LAMMPS!")
		else:
			try:
				self.lmp = lammps(cmdargs=cmdargs, comm=self.comm)
			except:
				raise ValueError("LAMMPS must be compiled with MPI support!")
		
		# Read LAMMPS input script
		lammps_input_script = self.params("lammps_input_script")
		assert os.path.exists(lammps_input_script)

		with open(lammps_input_script) as f:
			lammps_input_script = f.readlines()
		
		"""
		Overwriting
		"""
		# Replace input configuration if specified
		input_path = self.params("input_path",None)
		input_data = self.params("input_data",None)
		if input_path is not None and input_data is not None:
			input_data = os.path.join(input_path,input_data)
		if input_data is not None:
			assert os.path.exists(input_data)
			# Search for the line containing "read_data" and replace
			for i, line in enumerate(lammps_input_script):
				if len(line)>1:
					not_comment = line.strip()[0]!="#"
					if "read_data" in line.strip()[:14] and not_comment:
						break
			lammps_input_script[i] = "read_data "+input_data+"\n"
		
		# Replace pair style
		pair_style = self.params("pair_style",None)
		if pair_style is not None:
			assert os.path.exists(pair_style.strip().split(" ")[0])
			for i, line in enumerate(lammps_input_script):
				if len(line)>1:
					not_comment = line.strip()[0]!="#"
					if "pair_style" in line.strip()[:14] and not_comment:
						break
			lammps_input_script[i] = "pair_style "+pair_style+"\n"
		
		# Replace pair coeff
		pair_coeff = self.params("pair_coeff",None)
		if pair_coeff is not None:
			assert os.path.exists(pair_coeff.strip().split(" ")[0])
			for i, line in enumerate(lammps_input_script):
				if len(line)>1:
					not_comment = line.strip()[0]!="#"
					if "pair_coeff" in line.strip()[:14] and not_comment:
						break
			lammps_input_script[i] = "pair_coeff * * "+pair_coeff+"\n"
		
		# Replace mass
		mass = self.params("mass",None)
		if mass is not None:
			assert isinstance(mass,float) or isinstance(mass,int)
			for i, line in enumerate(lammps_input_script):
				if len(line)>1:
					not_comment = line.strip()[0]!="#"
					if "mass" in line.strip()[:14] and not_comment:
						break
			lammps_input_script[i] = f"mass * {mass}\n"
		
		lammps_input_script += ["thermo 1\nrun 0\n"]
		self.command("\n".join(lammps_input_script))

	def command(self, cmd_str):
		"""
			Run lammps command. Can catch errors here
		"""
		self.lmp.commands_string(cmd_str)
	
	def set_dummy_potential(self):
		cutoff = self.lmp.extract_pair("cutoff")
		print(cutoff)
		pass

	def get_natoms(self):
		"""Get number of atoms"""
		return self.lmp.get_natoms()

	def get_positions(self):
		"""Get atom positions"""
		return np.ctypeslib.as_array(self.lmp.gather("x",1,3)).copy()

	def run_nve(self,steps,safe=False):
		if safe:
			self.command(f"run {steps}")
		else:
			self.command(f"run {steps} pre no post no")
	
	def set_nve(self):
		self.command("fix seal all nve")
		self.nve_set = True
	
	def set_positions(self,x=None,dx=None,safe=False,velocity=False):
		"""Get atom positions"""
		assert not (x is None and dx is None), "x or dx must be provided!"
		if x is None:
			x = self.x_0.flatten() + dx.flatten()
		self.x = x.reshape((-1,3))
		self.lmp.scatter("x", 1, 3, np.ctypeslib.as_ctypes(x))
		if velocity:
			v_0 = np.zeros_like(x).flatten()
			self.lmp.scatter("v", 1, 3, np.ctypeslib.as_ctypes(v_0))
		if self.verbose and safe and self.rank==0:
			print("Running with neighbor list check")
		self.command("run 0" if safe else "run 0 pre no post no")
		
	def isosurface_sample(self, sample_keys):
		"""Sample from the isosurface"""
		return super().isosurface_sample(sample_keys)

	def get_mass(self):
		"""Get atom masses"""
		return float(self.lmp.extract_atom("mass")[1])

	def get_cell(self):
		boxlo,boxhi,xy,yz,xz,periodicity,box_change = self.lmp.extract_box()
		self.C = np.zeros((3,3))
		for cell_j in range(3):
			self.C[cell_j][cell_j] = boxhi[cell_j]-boxlo[cell_j]
		self.C[0][1] = xy
		self.C[0][2] = xz
		self.C[1][2] = yz
		return self.C
	
	def set_cell(self,C):
		self.command("change_box all triclinic")
		self.command(f"change_box all x final 0.0 {C[0][0]} y final 0.0 {C[1][1]} z final 0.0 {C[2][2]} xy final {C[0][1]} yz final {C[1][2]} xz final {C[0][2]} remap units box\nrun 0")
		self.C = C

	def set_constants(self):
		"""Set constants for the simulation"""
		self.C = self.get_cell()
		self.strain = 0.0
		self.zero_energy = self.get_potential_energy(offset=False)
		self.zero_descriptor = self.get_descriptor()
		
		self.x_0 = self.get_positions()
		self.shell_mat,self.shell_num = self.neighbor_shells()
		self.N_0 = self.get_natoms()
		self.mass = self.get_mass()
		self.constants = {
			'C_ref': self.C.copy(),
			'x_ref': self.x_0.copy(),
			'N': self.N_0,
			'mass': 1.0*self.mass,
			'shell_num' : self.shell_num.copy(),
			'x_0': self.x_0.copy(),
			'C_0': self.C.copy(),
			'E_0' : 1.0*self.zero_energy,
			'D_0' : self.zero_descriptor,
			'V_0' : np.linalg.det(self.C)
		}
		if self.compute_covar:
			self.zero_descriptor_tensor = self.get_per_atom_descriptor_covariance_tensor()
			self.constants['D_tensor_0'] = self.zero_descriptor_tensor.copy()

		
	def update_constants(self,extra=None):
		self.constants['C_0'] = self.C.copy()
		self.constants['V_0'] = np.linalg.det(self.C)
		self.constants['E_0'] = 1.0 * self.zero_energy
		self.constants['D_0'] = self.zero_descriptor.copy()
		self.constants['x_0'] = self.x_0.copy()
		if self.compute_covar:
			self.constants['D_tensor_0'] = self.zero_descriptor_tensor.copy()
		
		if not extra is None:
			assert isinstance(extra, dict)
			self.constants.update(extra)

	def rescale_box(self, strain):
		
		"""Rescale simulation box"""
		if abs(strain - self.strain) < 1e-12:
			return # do nothing
		
		if self.rank==0:
			print(f"\n\t\t\t\tRescaling box to {100.*strain:.3g}% strain")
		# reset all to zero
		self.reset(strain=True)
		self.strain = strain
		C = (1.0+strain) * self.C
		self.set_cell(C)
		self.zero_energy = self.get_potential_energy(offset=False)
		self.zero_descriptor = self.get_descriptor()
		if self.compute_covar:
			self.zero_descriptor_tensor = self.get_per_atom_descriptor_covariance_tensor()
		self.update_constants()
	
	def set_reference_volume(self, V=None):
		
		self.C = self.get_cell()
		self.N_0 = self.get_natoms()
		if not V is None:
			V_old = np.linalg.det(self.C)/self.N_0
			strain = (V/V_old)**(1.0/3.0) - 1.0
			C = (1.0+strain) * self.C
			self.constants['C_ref'] = C.copy()
			self.set_cell(C)
		
	def reset(self,strain=True):
		"""Reset atom positions"""
		if strain:
			self.set_cell(self.constants['C_ref'])
			self.set_positions(self.constants['x_ref'],safe=True,velocity=True)
			self.strain = 0.0
		else:
			self.x = self.x_0.copy()
			self.set_positions(self.x,safe=True,velocity=True)
		self.update_constants()
		
	def calculate_hessian_modes(self):

		""" 
			Calculate Hessian matrix, load and diagonalize
		"""
		self.command("dynamical_matrix all regular 0.00001"
						 			f" file {self.hessian_file}")
		if self.comm is not None:
			self.comm.Barrier()
		if self.size>self.local_size and self.rank==0:
			print(f"""\t\t\t\tMPI DANGER-
			Possible but dangerous to perform Hessian calculation in parallel!
			In general it is advised to run the Manager.test_run() function in serial on e.g. the login node, which will prepare for the full MPI run.""")

		# Currently MPI safe, but could be an issue
		H = np.loadtxt(self.hessian_file)
		Hlen = int(3 * np.sqrt(len(H) / 3))
		H = H.reshape((Hlen, Hlen)) * self.mass
		H -= np.diag(H.sum(1))
		return spla.eigh(H)

	def get_descriptor(self):
		"""Extract descriptor compute"""
		D = self.lmp.numpy.extract_compute(self.compute_name, \
				LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR).copy()
		return D
	def get_per_atom_descriptor_covariance_tensor(self):
		"""
		Extract the covariance tensor of a per-atom descriptor 
		with it's neighbors
		"""
		D_vec = np.ctypeslib.as_array(self.lmp.gather(\
					f"c_{self.vector_compute_name}",1,D.size)\
				).copy().reshape((-1,D.size))
		# per-atom
		D_per_atom = np.array([np.einsum('ia,jb,ij->ab',D_vec,D_vec,s) \
			for s in self.shell_mat])
		D_per_atom /= self.shell_num[:,None,None]
		return D_per_atom
	

	def get_potential_energy(self, offset=True, safe=True):
		"""Extract energy compute"""
		if safe:
			self.command("run 0")
		E = 1.0 * self.lmp.get_thermo("pe") / self.N_0
		if offset:
			E -= 1.0 * self.zero_energy
		return E
	
	def get_kinetic_energy(self,safe=False):
		"""Extract kinetic energy"""
		if safe:
			self.command("run 0")
		return 1.0 * self.lmp.get_thermo("ke") / self.N_0
	
	def get_total_energy(self,offset=True,safe=True):
		"""Extract total energy"""
		E = self.get_potential_energy(offset,safe)
		E += self.get_kinetic_energy(safe)
		return E

	def displacement_from_x0(self, x=None):
		"""Calculate displacement from initial positions"""
		if x is None:
			v = (self.x.flatten() - self.x_0.flatten()).reshape((-1, 3))
		else:
			v = (x.flatten() - self.x_0.flatten()).reshape((-1, 3))
		v -= np.floor(v @ np.linalg.inv(self.constants['C_0']) + 0.5) @ self.constants['C_0']
		return v.flatten()
	
	def neighbor_shells(self, shells=None):
		if shells is None:
			shells = self.params("neighbor_shells", 14)
		"""Calculate displacement from initial positions"""
		x_0 = self.x_0.copy().reshape((-1,3))
		iC = np.linalg.inv(self.C)
		C = self.C.copy()
		pbc = lambda v: v - np.floor(v @ iC + 0.5) @ C
		# rounded distances
		pbcd = lambda i=0: \
        	np.round(np.linalg.norm(pbc(x_0 - x_0[i][None,:]),axis=1),3)
		# matrix of distances
		M = np.array([pbcd(i) for i in range(x_0.shape[0])])
		# unique distances for shells
		d = np.unique(M[0])
		shells = min(shells,len(d)-1)
		assert d[shells]-d[shells-1] > 0.01,"too many shells!"
		shell_mat = np.array([np.abs(M-d[i])<0.1 for i in range(shells)])
		shell_num = shell_mat.sum(-1).sum(-1)
		return shell_mat,shell_num    
	
	def displace_normal(self,mag=0.02):
		"""Normal displacement"""
		x = self.x_0.flatten()
		dx = np.random.normal(size=x.size)
		dx /= np.linalg.norm(dx)
		dx *= np.sqrt(mag*x.size/3.0)
		self.set_positions(x+dx,safe=True)
		
	def displacement_from_x0(self, x=None):
		"""Calculate displacement from initial positions"""
		if x is None:
			v = (self.x.flatten() - self.x_0.flatten()).reshape((-1, 3))
		else:
			v = (x.flatten() - self.x_0.flatten()).reshape((-1, 3))
		v -= np.floor(v @ np.linalg.inv(self.constants['C_0']) + 0.5) @ self.constants['C_0']
		return v.flatten()

	def target_energy(self, U,steps=None):
		"""
			Goal: create configuration with a given total energy U

			Have potential V and kinetic K, V = 0 initially
			
			Set K to temperature of T = U/(3/2 kB), then run NVE 

			We then check the energy to see if we are at U

			If not at U, scale the velocities so that K+V=U
		"""
		if not self.nve_set:
			self.set_nve()
		
		if steps is None:
			steps = 2*self.N_0
		if self.rank==0:
			print(f"\t\t\t\tNVE for {steps} steps")
		T = U/(self.kB * 1.5) * (1.00 + 1.0/self.N_0)

		seed = np.random.randint(10000, 100000)
		self.command(f"""
			reset_timestep 0
			velocity all create {T} {seed} dist gaussian
		""")
		self.run_nve(steps,safe=True)

		# find what temperature would give perfect match
		V = self.get_potential_energy(offset=True)
		# If K was at this temperature, we would have U=K+V
		T_req = (U-V)/(self.kB * 1.5)* (1.00 + 1.0/self.N_0)
		self.command(f"""
			reset_timestep 0
			velocity all scale {T_req} # scale to target T
		""")
		self.run_nve(10,safe=True)

	def close(self):
		"""Close LAMMPS instance"""
		self.lmp.close()
