import sys,yaml,os

class YAMLParser:
	def __init__(self,yaml_file="./input.yaml",verbose=False,**kwargs):
		"""
		Wrapper to load yaml file from command line

		Parameters
		----------
		yaml_file : None | os.PathLike[str], optional
			_description_, by default 'input.yaml'
		
		verbose : bool, optional
			output verbosity, by default `False`

		Raises
		------
		SyntaxError
			If yaml file has reading errors
		FileNotFoundError
			if `yaml_file` is not found
		"""
		if yaml_file is None:
			if self.verbose:
				print("\tDEFAULT: input.yaml")
				yaml_file = 'input.yaml'
		
		self.yaml_file = yaml_file
		self.verbose=verbose
		try:
			if self.verbose:
				print(f"\tTrying to open yaml file {yaml_file}")
			with open(yaml_file, 'r') as f:
				self.input_params = yaml.safe_load(f)
		except FileNotFoundError:
			raise FileNotFoundError(f"Cannot find {yaml_file}")
		
		if 'verbose' in self.input_params:
			self.verbose = bool(self.input_params['verbose'])
		self.input_params['verbose'] = self.verbose
		
		if kwargs:
			for kk in kwargs:
				self.input_params[kk] = kwargs[kk]
		
		if self.verbose:
			for kk in self.input_params:
				print(kk,self.input_params[kk])
	
	def has_key(self,key,options=None):
		if key in self.input_params:
			if options is None:
				return True
			else:
				if not self.input_params[key] in options:
					return False
		else:
			return False
	
	def __call__(self,key,default=-1):
		try:
			value = self.input_params[key]
			if self.verbose:
				print(f"\tReading {self.yaml_file} {key}:{value}")
		except:
			if default==-1:
				raise Exception(f"Could not find key {key}")
			else:
				value = default
				if self.verbose:
					print(f"\tReading DEFAULT {key}:{value}")
		return value