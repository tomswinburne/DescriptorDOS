import numpy as np
import pickle as pkl
import os
from itertools import product
from time import time
from mpi4py import MPI
from .MPIAgent import MPIAgent
from .TensorCompressor import TensorCompressor
from .Constants import eV,Boltzmann

class DDOSManager:
		""" 
			Manager class for running DOS calculations.
		"""
		def __init__(self,comm,yaml_file,seed=None,**kwargs):
			"""
				Initialize the DDOS Manager class.
				Parameters:
				-----------
				comm (MPI.Comm): The MPI communicator.
				
				yaml_file (str): Path to the YAML configuration file.
				
				seed (int, optional): Seed value for random number generation. Defaults to 13.

				kwargs (dict, optional): Additional keyword arguments, overwriting yaml_file parameters. Defaults to {}.
				
				Raises:
				-------
				AssertionError: If the YAML configuration file does not exist.
			"""
			assert os.path.exists(yaml_file)

			self.rank = comm.Get_rank()

			if seed is None:
				seed = int(np.round(time(),5))
			self.all_jobs = []
			self.comm = comm
			self.Agent = MPIAgent(	world=self.comm,\
									param_file=yaml_file,\
									setup=False,\
									seed=self.rank+seed,\
										**kwargs)
			
			self.params = self.Agent.params 
			self.combine = self.params("append_data",False)

			self.input_parameters = self.params("input_parameters",False)
			if not self.input_parameters:
				raise ValueError("input_parameters not found in yaml file!")
			self.jobs = self.create_jobs(self.input_parameters)
			
			# load data
			self.all_jobs = self.load_data(self.Agent.file_name)

			# finish setup of agent 
			self.Agent.finish_setup()



		def create_jobs(self,input_parameters):
			"""
				Create jobs from input parameters.
			"""
			# Create arrays for each parameter using np.linspace
			param_arrays = {}
			iso_surface_params = None
			for key, value in input_parameters.items():
				if 'IsoSurface' in key:
					iso_surface_params = value
				else:
					if isinstance(value[-1],int) and len(value)==3:
						start, stop, num = value
						param_arrays[key] = np.linspace(start, stop, num)
					else:
						param_arrays[key] = np.array(value)

			# Ensure IsoSurface is the last axis
			if iso_surface_params:
				value = iso_surface_params
				if isinstance(value[-1],int) and len(value)==3:
					start, stop, num = value
					param_arrays['IsoSurface'] = np.linspace(start, stop, num)
				else:
					param_arrays['IsoSurface'] = np.array(value)
				

			# Generate all combinations of parameter values
			param_combinations = product(*param_arrays.values())

			# Create jobs list with all parameter combinations
			jobs = []
			for combination in param_combinations:
				job = dict(zip(param_arrays.keys(), combination))
				jobs.append(job)

			if self.rank == 0:
				n = len(param_arrays.keys())
				axes = ", ".join(param_arrays.keys())
				print(f"\t\tTotal of {len(jobs)} jobs over {n} axes: {axes}\n")			
			return jobs

		def load_data(self,file_name):
			"""
				Load data from a pickle file on rank 0.
				Parameters:
				-----------
				file_name (str): Path to the pickle file.
				
				keys (list): List of keys required to be in the data.
				
				Returns:
				--------
				all_jobs (list): 
					List of jobs from the pickle file.
				
			"""
			if self.rank==0:
				all_jobs = []
				if os.path.exists(file_name):
					# load data	
					with open(file_name,"rb") as f:
						all_jobs = pkl.load(f)
					print(f"\t\t\tLoaded {file_name}\n")		
				else:
					all_jobs = []
			else:
				all_jobs = None 
			
			return all_jobs


		def check_seen(self,keys):
			"""
				Check if the target has been seen before.
				Parameters:
				-----------
				keys (dict): The key values to check
				
				Returns:
				--------
					seen (bool): True if the target has been seen
					seen_index (int): Index of the seen data in jobs list
			"""
			seen = False
			seen_index = None
			if self.rank==0:
				seen_index = 0
				for job in self.all_jobs:
					seen = True 
					for k,v in keys.items():
						seen = seen and np.isclose(job[k], v)
					if seen:
						break
					seen_index += 1
			
			seen = self.comm.bcast(seen, root=0)
			seen_index = self.comm.bcast(seen_index, root=0)
			return seen, seen_index

		def close(self):
			self.Agent.close()
		
		def print_header(self,current_job,total_jobs,target_keys,seen_before):
			if self.rank==0:
				print(f"\n\t\tJob {current_job} of {total_jobs} : ")
				print(f"\t\t\tStrain : {100*target_keys['Strain']:.3g}%",end="")
				for k in target_keys.keys():
					if k == 'Strain':
						continue
					else:
						print(f", {k} : {target_keys[k]:.4g}",end="")
				if seen_before:
					if self.combine:
						print(". Seen before, combining.")
					else:
						print(". Seen before, skipping.")
				else:
					print(".")
						
		def dump_uncompressed(self,uncompressed_samples,target_keys):
			if self.rank==0 and self.params("store_uncompressed_data",False):
				print("STORING UNCOMPRESSED DESCRIPTOR DATA")
				tag= "_uncompressed"
				for key,val in target_keys.items():
					tag += f"_{key}_{np.round(val,3)}"
				with open(os.path.splitext(self.Agent.file_name)[0] + tag+".pkl","wb") as f:
					pkl.dump(uncompressed_samples,f)
					

		def print_results(self,collated_samples,t):
			if self.rank==0:
				tot_kb = 1.5 * Boltzmann / eV
				
				accuracy = np.round(\
					collated_samples['IsoSurfaceMeasured_mean'].mean(),3)
				std = np.round(\
					np.sqrt(collated_samples['IsoSurfaceMeasured_var'].mean()),3)
				
				alpha_accuracy = np.round(\
					collated_samples['AlphaMeasured_mean'].mean(),3)
				alpha_std = np.round(\
					np.sqrt(collated_samples['AlphaMeasured_var'].mean()),3)
				
				EqTemperature = np.round(\
					collated_samples['EqiTemperature_mean'],3)
				T_std = np.round(\
					np.sqrt(collated_samples['EqiTemperature_var']),3)
				
				print(f"\n\t\t\tTime: {time()-t:.4g}s, "
					f"IsoSurface: {accuracy} +/- {std}, alpha: {alpha_accuracy} +/- {alpha_std}, T_eq: {EqTemperature} +/- {T_std}K\n")
		
		def compress_data(self,uncompressed_samples,seen_samples=None):
			"""
				Compress the data.
				Parameters:
				-----------
				uncompressed_samples (dict): Uncompressed samples.
				seen_samples (dict, optional): Seen samples. Defaults to None.
			"""
			compressed_samples = None
			if self.rank==0:
				# Compress descriptor data, using seen_samples
				new_calls = uncompressed_samples['D'].shape[0]
				old_calls = seen_samples['calls'] if seen_samples is not None else 0
				print(f"\t\t\t\tAdding {new_calls} results: {old_calls}",end="")
				
				AuxillaryField = None 
				if self.params("isosurface","Hessian") == 'Kinetic':
					AuxillaryField = uncompressed_samples['K']

				compressed_samples = TensorCompressor(uncompressed_samples['D'],self.Agent.N,
										CompressedData=seen_samples,
										AuxillaryField=AuxillaryField,
										bins = self.params("histogram_bins",100),
										bootstrap_ensemble_size= self.params("bootstrap_ensemble",10),
										threshold_percentile = self.params("histogram_percentile",5.0))
				
				compressed_samples['calls'] = new_calls

				for k in ['IsoSurfaceMeasured','E','K','EqiTemperature','AlphaMeasured']:
					compressed_samples[f"{k}_mean"] = uncompressed_samples[k].mean()
					compressed_samples[f"{k}_var"] = uncompressed_samples[k].var()
				
				if 'D_tensor' in uncompressed_samples:
					compressed_samples['D_tensor_mean'] = uncompressed_samples['D_tensor'].mean(0)
					compressed_samples['D_tensor_mean'] -= np.outer(compressed_samples["D_mean"],compressed_samples["D_mean"])
					ev = np.array(\
						[np.linalg.eigvals(r).max() for r in compressed_samples['D_tensor_mean']])
				
				if seen_samples is not None:
					new_w = 1.0 * new_calls/(new_calls+old_calls)
					old_w = 1.0 * old_calls/(new_calls+old_calls)
					for k in ['IsoSurfaceMeasured','E','EqiTemperature','AlphaMeasured','D_tensor','D']:
						for mv in ['mean','var']:
							key = f'{k}_{mv}' 
							if key in compressed_samples:
								compressed_samples[f'{k}_{mv}'] = compressed_samples[f'{k}_{mv}']*new_w \
													+ seen_samples[f'{k}_{mv}']*old_w
					compressed_samples['calls'] += old_calls
				print(f" -> {compressed_samples['calls']}")
				
				# Update results with new averages
				compressed_samples['IsoSurface'] = uncompressed_samples['IsoSurface'].mean()
				compressed_samples['Strain'] = uncompressed_samples['Strain'].mean()

			return compressed_samples
			

		def test_run(self):
			"""
				Run the DOS calculations just for one call
				Can prepare for full run after testing
			"""
			for current_job, job in enumerate(self.jobs):
				self.print_header(current_job+1,len(self.jobs),job,False)
				samples = self.Agent.run(job=job,calls=1)

			
		def run(self):
			"""
				Run the DOS calculations.
			"""
			for current_job, job in enumerate(self.jobs):
				
				seen, seen_index = self.check_seen(job)
				if seen and self.rank==0:
					seen_samples = self.all_jobs[seen_index]
				else:
					seen_samples = None
				
					
				self.print_header(current_job+1,len(self.jobs),job,seen)
					
				t = time() if self.rank==0 else None
				if seen and (not self.combine):
					continue
				
				# Generate samples on all ranks
				samples = self.Agent.run(job=job)
				
				# Collate samples on rank 0 with MPI Gather
				# TODO batch this across processors
				uncompressed_samples, sample_count = self.Agent.collate(samples)

				# Dump uncompressed data if requested (NOT ADVISED! TESTING ONLY!)
				self.dump_uncompressed(uncompressed_samples,job)
				
				# Compress data and merge with seen data if present
				compressed_samples = self.compress_data(uncompressed_samples,seen_samples)
				
				# remove uncompressed data
				del uncompressed_samples

				# Store all other fields in a list
				if self.rank==0:
					constants = self.Agent.get_reference_data()
					compressed_samples.update(constants)
				
				# Print results
				self.print_results(compressed_samples,t)
				
				if self.rank==0:
					if seen:
						# If seen before, update data
						self.all_jobs[seen_index] = compressed_samples
					else:
						# If not seen before, add data
						self.all_jobs += [compressed_samples]
					
					# running total... not super efficient but not a big deal
					with open(self.Agent.file_name,"wb") as f:
						pkl.dump(self.all_jobs,f)
					
					t = time()
			if self.rank==0:
				print(f"\n\n\t\t------------------------"\
					"---------------------------------------\n\n\t\t"\
					f"FINISHED DDOS SAMPLING\t\t\tData file : {self.Agent.file_name}\n")
					