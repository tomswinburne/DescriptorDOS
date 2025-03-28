import itertools,os,pickle,time
import numpy as np
from .Constants import hbar,eV,atomic_mass,Boltzmann

class EinsteinDisplacer:
    """
    EinsteinDisplacer class for generating Gaussian displacements and preparing vibrational modes.
    
    Methods
    ----------
    __init__(worker)
        Initializes the EinsteinDisplacer class with the given worker and optional preparation flag.

    __call__(isosurface_temperature=0.1,safe=False)
        Generates Gaussian displacement and returns a dictionary with displacement vector, mean squared displacement, harmonic energy, and max squared displacement.

    prepare(worker)
        Determines the energy scale by displacing atoms randomly and calculating the mean vibrational frequency.
    """
    def __init__(self,worker, run_prepare=True):
        """
        Initializes the Displacers class.

        Parameters
        ----------

        worker:  Worker
            The Worker class instance.
        
        run_prepare: bool
            Flag to run the prepare method. Default is True.

        Attributes
        ----------
        read_modes : bool
            Indicates if modes should be read from the worker.
        mass : float
            Mass of the atom extracted from the worker.
        rank : int
            Rank of the worker.
        omega_mean : float
            Mean value of omega, initialized to 1.0.
        U_0 : float
            Reference energy for the isosurface, initialized to 1.0. 
        x_0 : numpy.ndarray
            Initial positions of atoms reshaped to (-1, 3).
        N_0 : int
            Number of atoms.
        hbar : float
            Reduced Planck constant in eV*ps.
        eV_AA : float
            Conversion factor from eV to amu/ps^2.
        constants : dict
            Dictionary to store constants.
        """
        
        self.mass = 1.0*worker.mass
        
        self.rank = 1*worker.rank
        self.omega_mean = 1.0
        self.strain = worker.strain
        
        # Can also access supercell etc from worker if needed
        self.x_0 = worker.x_0.copy().reshape((-1,3))
        self.N_0 = worker.N_0
        self.dx = np.zeros_like(self.x_0)
        
        self.hbar = hbar / eV * 1e12 # eV ps
        self.eV_AA = eV / (atomic_mass * (1e-10)**2 / (1e-12)**2)
        self.kB = Boltzmann / eV # eV/K
        
        self.U_0 = 1.5 * self.kB * 1000.0
        
        
        self.constants = {}
        if run_prepare:
            self.prepare(worker)
        
        
    def __call__(self,worker,isosurface_temperature=0.1,safe=False):
        """
        Generate Gaussian Displacement.

        Args:
        ----
            worker: LAMMPSWorker instance

            isosurface_temperature (float, optional): Isosurface Value. Default is 0.1
        
        Returns
        -------
        results: A dictionary containing the following keys:
            - 'IsoSurfaceMeasured' (float): empirical isosurface value
            - 'EqiTemperature' (float): isosurface equipartition temperature
            - 'MaxSquaredDisplacement' (float): The max absolute displacement.
        """ 

        # sample on unit sphere
        dv = np.random.normal(0,1.0,size=3*self.N_0)
        dv /= np.linalg.norm(dv)

        # Rotate to level set of alpha in basis B_l
        # |dv.B_l| = 1, dx = a_l * v_l @ B_l
        # alpha = ln|dX.H.dX/2N| = ln< nu_l a^2_l >
        # => < nu_l a^2_l > = exp(alpha) => a_l = rt(exp(alpha)/nu_l)
        # (U_0/l^2)<dx> = U_0 exp(alpha)
        rsq = isosurface_temperature * self.U_0 / (self.omega_mean*0.5)
        dv *=  np.sqrt(rsq * self.N_0)
        self.dx = dv.copy()
        # Do we need this now?
        
        # PER ATOM
        sample_rsq = (dv**2).sum() / self.N_0
        H_e = sample_rsq * self.omega_mean / 2.0
        H_r = worker.get_potential_energy(offset=True)
        worker.set_positions(dx=self.dx,safe=safe)

        return {'IsoSurfaceMeasured': H_e / self.U_0,
                'AlphaMeasured' : np.log(H_e/self.U_0)-self.alpha_shift,
                'EqiTemperature': H_r / 1.5 / self.kB,
                'MaxSquaredDisplacement':np.abs(self.dx).max()}
    
    def scale(self,worker,factor=1.01,safe=False):
        self.dx *= factor
        worker.set_positions(dx=self.dx,safe=safe)

    def prepare(self,worker):
        """
        Prepares the vibrational modes for the given worker by displacing atoms 
        randomly and calculating the mean vibrational frequency.
        Parameters:
        worker (LAMMPSWorker): A LAMMPSWorker instance
        
        This method performs the following steps:
        1. Displaces all atoms randomly within a specified magnitude.
        2. Runs a zero-step simulation to update the system state.
        3. Extracts the energy of the system.
        4. Resets the worker to its initial state.
        5. Calculates the mean vibrational frequency (omega_mean) based on the 
            extracted energy and displacement magnitude.
        6. Computes a constant F_A based on the system's properties.
        7. Sets the constants attribute with F_A, F_B, and log_det values.
        8. If the rank is 0, prints information about the zero modes, harmonic 
            free energy, and other related constants.
        Notes:
        - The displacement magnitude is fixed at 0.02.
        - The random seed for displacement is set to 123.
        - The energy extraction does not reset the energy to zero.
        - The constants dictionary includes F_A, F_B, and log_det.
        """
        
        # uniform in +-0.02 => 
        if np.abs(worker.strain-self.strain)>1.0e-4 or self.omega_mean==1.0:
            self.strain = worker.strain
            mag = 0.02
            # |dx|^2 = N * mag
            # E = N * mag * omega_mean / 2.0
            # omega_mean = 2.0 * E / mag / N
            worker.displace_normal(mag)
            E = worker.get_potential_energy(offset=True)
            self.omega_mean = 2.0 * E / mag
            worker.reset(strain=False)
            F_A = np.sqrt(self.eV_AA * 1.0 / self.mass) * self.hbar
            self.alpha_shift = np.log(self.omega_mean/2.0/self.U_0)
        
            self.constants = {  'F_A':F_A, 'F_B':-3.0, \
                            'log_det':0.0, 'U_0':self.U_0,\
                            'alpha_shift':self.alpha_shift,\
                            'flavor':'Einstein'}

            if self.rank==0:
                print(f"\n\t\t\tZero Modes = 0/{3*self.N_0}, "\
                    f"<ln|w/<w>|> = 0, "\
                    f"F_H = {F_A:.4g}*kT -3kT*ln|kT|\n")
        
class KineticDisplacer(EinsteinDisplacer):
    """
    KineticDisplacer class is responsible for displacing atoms kinetically in a molecular dynamics simulation. It inherits from EinsteinDisplacer and provides methods to prepare the system, and generate NVE displacements.

    __init__(self, worker, run_prepare=True):
        Initializes the KineticDisplacer with the given worker.

    prepare(self, worker):
        Prepares the worker by setting initial positions, resetting the worker, sealing the energy, and extracting the initial energy.
    
    __call__(self, worker, isosurface_temperature=0.10, steps=2,safe=False):
        Displaces the system kinetically by targeting a specific energy and running the simulation for a given number of steps. Returns a dictionary with the expected alpha value, reference energy, and maximum squared displacement.
    """

    def __init__(self,worker,run_prepare=True):
        super().__init__(worker,run_prepare=False)
        self.U_0 = 3.0 * self.kB * 1000.0
        self.prepare(worker)
    
    def prepare(self,worker):
        # check for changes in strain?
        # Monitor, but nothing to do here
        self.strain = worker.strain
        worker.set_nve()
        self.alpha_shift = 0.0
        self.constants = {'F_A':0.0, 'F_B':-3.0, \
            'log_det':0.0, \
            'alpha_shift' : self.alpha_shift, \
            'U_0':self.U_0,'flavor':'Kinetic'}
    
        
        # nothing
    def __call__(self,worker,isosurface_temperature=0.1,steps=2,safe=False):

        U = self.U_0*isosurface_temperature
        Epre = worker.get_total_energy(offset=True)
        if np.abs(Epre/U-1.0)>0.05:
            worker.target_energy(U)
        
        worker.run_nve(steps=steps,safe=safe)
        V = worker.get_potential_energy(offset=True)
        K = worker.get_kinetic_energy()
        E = V + K
        return {'IsoSurfaceMeasured': E / self.U_0,
                'AlphaMeasured' : np.log(E/self.U_0)-self.alpha_shift,
                'KineticEnergy' : K,
                'EqiTemperature': E/3.0/self.kB,
                'MaxSquaredDisplacement':0.0}


class HessianDisplacer(EinsteinDisplacer):
    """
    HessianDisplacer class is responsible for generating and managing Hessian modes for a given worker. It inherits from EinsteinDisplacer and provides methods to set normal mode files, generate and save modes, and prepare modes for further calculations.

    Methods:
    ----------
        __init__(self, worker):
            Initializes the HessianDisplacer with the given worker. Prepares modes and sets regularization parameter.
        set_normal_mode_file(self, worker):
            Sets the normal mode file path based on worker parameters.
        make_and_save_modes(self, worker):
            Generates the Hessian matrix, computes its eigenvalues and eigenvectors, and saves them to a file.
        
        prepare(self, worker):
            Loads mode data from a file if available. If not, generates and saves modes. Stores various mode-related information for further calculations.
        
        __call__(self, isosurface_temperature=1.0,safe=False):
            Samples on a hypersurface and returns a dictionary containing displacement vector, mean squared displacement, harmonic energy, and maximum squared displacement.
    
    Attributes:
    ----------
        nm_file (str): Path to the normal mode file.
        omega (ndarray): Eigenvalues of the Hessian matrix.
        modes (ndarray): Eigenvectors of the Hessian matrix.
        stable (ndarray): Boolean array indicating stable modes.
        num_modes (int): Number of stable modes.
        omega_mean (float): Mean of the stable eigenvalues.
        modes_stable (ndarray): Transposed eigenvectors of stable modes.
        nu_stable (ndarray): Normalized stable eigenvalues.
        mode_amplitude (ndarray): Amplitude of the modes.
        constants (dict): Dictionary containing constants F_A, F_B, and log_det.
    """
    def __init__(self,worker,run_prepare=True):
        super().__init__(worker,run_prepare=False)
        self.nm_data=None

        if run_prepare:
            self.prepare(worker) 
    
    def set_normal_mode_file(self,worker):
        nm_file = worker.params("hessian_data","HessianModes")
        dump_path = worker.params("dump_path","./")
        hessian_path = worker.params("hessian_path",dump_path)
        self.nm_file = os.path.join(hessian_path,nm_file)

    def load_modes(self):
        """ 
        Loads the normal mode data from the specified file.
        Parameters:
            None
        Returns:
            None
        """


        read_modes = os.path.exists(self.nm_file) 
        read_modes *= self.nm_data is None
        if read_modes:
            try:
                if self.rank==0:
                    print(f"\t\t\t\tLoading Hessian data from {self.nm_file}")
                # dictionary indexed by strain
                with open(self.nm_file,"rb") as f:
                    self.nm_data = pickle.load(f)
            except Exception as e:
                raise IOError(e)
        else:
            if self.nm_data is None:
                self.nm_data = {}
                return False 
        
        # which strains are present
        strain_key = int(1000.*self.strain)
        strain_keys = [int(s) for s in self.nm_data.keys()]
        if strain_key in strain_keys:
            if self.rank==0:
                print(f"\t\t\t\tFound Hessian for {0.1*strain_key:.4g}% strain")
            self.omega,self.modes = self.nm_data[strain_key]
            has_modes = self.omega.size == self.x_0.size
        else:
            if self.rank==0:
                print(f"\t\t\t\tNo Hessian for {0.1*strain_key:.4g}% strain")
            has_modes = False 
        return has_modes
        
    def make_and_save_modes(self, worker):
        """ 
        Generates the Hessian matrix, computes its eigenvalues and eigenvectors,
        and saves the results to a file.
        Parameters:
            worker (LAMMPSWorker): A LAMMPSWorker instance 
        Returns:
            None
        """
        if self.rank==0:
            t = time.time()
            print("\t\t\t\tGenerating Hessian...")
        self.omega, self.modes = worker.calculate_hessian_modes()
        strain_key = int(1000.*self.strain)
        self.nm_data[strain_key] = [self.omega,self.modes]
        if self.rank==0:
            print(f"\t\tHessian made in {time.time()-t:.4g}s")
            with open(self.nm_file,"wb") as f:
                pickle.dump(self.nm_data,f)
            print(f"\t\tSaved to {self.nm_file}")
            
            
    def prepare(self,worker):
        """
        Prepares the necessary data for the displacement calculations.
        This method performs the following steps:
        1. Sets the normal mode file using the provided worker.
        2. Checks if the normal mode file exists and reads the modes if available.
        3. If the modes are not available, it generates and saves the modes.
        4. Stores the necessary information for the alpha function, including:
            - Stability of the modes.
            - Number of stable modes.
            - Mean of the stable frequencies.
            - Transposed stable modes.
            - Normalized stable frequencies.
        5. Converts frequencies to hbar and calculates constants F_A and F_B.
        6. Calculates the mode amplitude and log determinant for the hypersphere.
        7. Prints relevant information if the rank is 0.
        
        Parameters:
        ----------
        worker (LAMMPSWorker): A LAMMPSWorker instance

        Raises:
        ------
        IOError: If there is an issue reading the normal mode file.
        AssertionError: If the size of omega does not match the size of x_0.
        """
        
        # if strain has changed we reload modes
        remake = np.abs(worker.strain-self.strain)>1.0e-4 

        remake += self.nm_data is None
        if remake:
            self.set_normal_mode_file(worker)
            self.strain = worker.strain
            has_modes = self.load_modes()
            if not has_modes:
                self.make_and_save_modes(worker)        
        # we now have self.omega and self.modes

        # Store all information needed for alpha function
        self.stable = self.omega>1.0e-6
        self.num_modes = self.stable.sum()
        self.omega_mean = self.omega[self.stable].mean()
        self.modes_stable = self.modes.T[self.stable] # NB transpose!
        self.nu_stable = self.omega[self.stable] / self.omega_mean

        r_m = float(self.num_modes) / self.N_0 # -> 3 from below as N->inf

        # convert frequencies to hbar
        h_w = np.sqrt(self.eV_AA*self.nu_stable/self.mass) * self.hbar
        F_A = np.log(h_w).mean() * r_m
        # F_A = -F_B * (0.5*log_det) + const
        F_B = -r_m
        
        # for hypersphere
        self.mode_amplitude = 1.0 / np.sqrt(self.nu_stable)
        log_det = np.log(self.nu_stable).mean() # <=0 as <nu>==1
        
        self.alpha_shift = np.log(self.omega_mean/2.0/self.U_0)
        # F_H = F_A * kT - F_B * kT * ln|kT|
        # Mass m-> m' : <ln|hw|> -> <ln|hw|> + ln|m'/m|
        self.constants = {  'F_A':F_A,'F_B':F_B, \
                            'log_det':log_det,\
                            'U_0':self.U_0,\
                            'omega_mean':self.omega_mean,\
                            'alpha_shift':self.alpha_shift,\
                            'flavor':'Hessian'}

        if self.rank==0 and remake:
            # f"<ln|w/<w>|> = {log_det:.3g}, "\
            print(f"\n\t\t\tZero Modes = {(~self.stable).sum()}/{3*self.N_0}, "\
                f"F_H = {F_A:.4g}*kT -{abs(F_B):.4g}kT*ln|kT|\n")
            
    def __call__(self, worker, isosurface_temperature = 0.1,safe=False):
        """
        Generates a displacement vector based on a random sample on the unit sphere, scaled by the mode amplitude and the given exponential alpha parameter.

        Args:
        ----
            worker: LAMMPSWorker instance

            isosurface_temperature (float, optional): Isosurface `temperature'. Default is 0.1
        
        Returns
        -------
        results: A dictionary containing the following keys:
            - 'IsoSurfaceMeasured' (float): empirical isosurface value
            - 'EqiTemperature' (float): isosurface equipartition temperature
            - 'MaxSquaredDisplacement' (float): The max absolute displacement.
        """ 
        # sample on unit sphere
        dv = np.random.normal(0,1.0,size=self.num_modes)
        dv /= np.linalg.norm(dv)

        # Rotate to level set of alpha in basis B_l
        # |dv.B_l| = 1, dx = a_l * v_l @ B_l
        # alpha = ln|dX.H.dX/2N| = ln< nu_l a^2_l >
        # => < nu_l a^2_l > = exp(alpha) => a_l = rt(exp(alpha)/nu_l)
        # (U_0/l^2)<dx> = U_0 exp(alpha)
        rsq = isosurface_temperature * self.U_0 / (self.omega_mean*0.5)
        dv *= self.mode_amplitude * np.sqrt(rsq * self.N_0)
        self.dx = dv@self.modes_stable
        # Do we need this now?
        
        # PER ATOM
        sample_rsq = (self.nu_stable * dv**2).sum() / self.N_0
        H_e = sample_rsq * self.omega_mean / 2.0
        worker.set_positions(dx=self.dx,safe=safe)
        

        return {'IsoSurfaceMeasured': H_e / self.U_0,
                'AlphaMeasured' : np.log(H_e/self.U_0)-self.alpha_shift,
                'EqiTemperature': H_e / 1.5 / self.kB,
                'MaxSquaredDisplacement':np.abs(self.dx).max()}


class KineticHessianDisplacer(HessianDisplacer):
    """
    KineticHessianDisplacer class is responsible for displacing atoms kinetically, then measuring 
    the isosurface defined by a Hessian modes. 
    It inherits from HessianDisplacer and provides methods to prepare the system, and generate NVE displacements.

    __init__(self, worker):
        Initializes the KineticHessianDisplacer with the given worker. Prepares modes and sets regularization parameter.

    prepare(self, worker):
        Prepares the worker by setting initial positions, resetting the worker, sealing the energy, and extracting the initial energy.
    
    __call__(self, worker, isosurface_temperature=0.1, steps=2,safe=False):
        Displaces the system kinetically by targeting a specific energy and running the simulation for a given number of steps. Returns a dictionary with the expected alpha value, reference energy, and maximum squared displacement.
    """

    def __init__(self,worker,run_prepare=True):
        super().__init__(worker,run_prepare=True)
        worker.set_nve()
        self.constants.update({'flavor':'KineticHessian'})
        
    def get_harmonic_energy(self,worker):
        """
            Obtain current alpha value
        """
        x = worker.get_positions()
        dx = worker.displacement_from_x0(x) 
        dv = dx @ self.modes_stable.T
        sample_rsq = (self.nu_stable * dv**2).sum() / self.N_0
        return sample_rsq * self.omega_mean / 2.0
        
        # nothing
    def __call__(self,worker,isosurface_temperature=0.1,steps=2,safe=False):

        U = 2.0*self.U_0*isosurface_temperature  # kinetic energy in equipartition
        Epre = worker.get_potential_energy(offset=True)
        
        if np.abs(Epre/U-1.0)>0.05:
            worker.target_energy(U)
        worker.run_nve(steps=steps,safe=safe)
        
        E = self.get_harmonic_energy(worker) # harmonic energy
        return {'IsoSurfaceMeasured': E / self.U_0,
                'AlphaMeasured' : np.log(E/self.U_0)-self.alpha_shift,
                'EqiTemperature': E /1.5/self.kB,
                'MaxSquaredDisplacement':np.abs(self.dx).max()}


Displacers = {"Einstein":EinsteinDisplacer,
              "Hessian":HessianDisplacer,
              "Kinetic":KineticDisplacer,
              "KineticHessian":KineticHessianDisplacer}
