import os,sys
sys.path.insert(0,'../DescriptorDOS/')
from mpi4py import MPI
from DescriptorDOS import DDOSManager 
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run DDOSManager with specified element and structure.')
parser.add_argument('--structure', type=str, default='a15', help='Structure to use (e.g., bcc, a15)')

args = parser.parse_args()
size = MPI.COMM_WORLD.Get_size()

structure = args.structure

assert structure in ['bcc','a15'], f"Structure {structure} not recognized. Please use 'bcc' or 'a15'."

# reference volume for phase
V_target = {'bcc': 16.16228768,'a15':16.45160002}

# Define the mass and potential for the element
mass = 183.84
ref_path=f'./W/'
potential = f"X_W.snapcoeff"

# In this case, we only sample at reference volume
strain_list = [0.0] # could have e.g. [0.0,0.01,0.02] for 0%-2% a strain sweep

# Isosurface "temperature" (1.0 = similar 1000K)
if structure == 'bcc':
        # e.g. around 500K-4000K range, appropriate for bcc W
        isosurface_temperature_list = list(np.linspace(0.5,4.0,35))
else:
        # if a15, use a narrower range of isosurface values
        isosurface_temperature_list = list(np.linspace(0.5,3.0,25))

if size==1:
        # if on one proc, assume for testing
        ncalls = 100
        dump_path = 'temp-to-delete/' # to not store sampling data
        isosurface_temperature_list = [0.5,3.0]
else:
        ncalls = 300 * size # 300 samples per proc
        dump_path = os.path.join(ref_path,f'output/')

for isosurface in ['Hessian','Kinetic']:
        
        # Set up the DDOSManager
        DDOS_Manager = DDOSManager(comm=MPI.COMM_WORLD,
        
        # Type of isosurface to compute: 'Kinetic' or 'Hessian'
        isosurface=isosurface,

        # Path and filename for extra configuration options (overwritten)
        yaml_file="configuration-fine.yaml",

        # Path and filename for LAMMPS input script
        lammps_input_script=os.path.join(ref_path,"in.lammps"),
        
        # Path and filename for output data
        dump_path=dump_path,
        dump_file=f'DDOSData-W-{structure}-{isosurface}',
        
        # number of descriptor samples per isosurface
        default_calls=ncalls,
        
        # Reference volume for phase (e.g. bcc, a15) from which strain is calculated
        V_target = V_target[structure],

        # path and filename for input structure
        input_path=os.path.join(ref_path,'configurations/'), 
        input_data= f'W-{structure}.dat', 
        
        # path and filename for Hessian data (if used)
        hessian_path=os.path.join(ref_path,'hessians/'),
        hessian_data= f'W-{structure}.pkl', 
        
        # if True, will print out more information
        verbose=False, 
        
        # if True, will store uncompressed data NOT recommended
        store_uncompressed_data=False, 
        
        # percentile threshold for histogram - ignore tails for better binning.
        percentile_threshold=5, 
        
         # see above for details
        input_parameters={'IsoSurface' : isosurface_temperature_list,'Strain': strain_list},
        
        # if True, will compute per-atom covariance matrix. Experimental
        compute_covar=False, 
        
        # if True, will append data to existing file
        append_data=True, 
        
        # pair_coeff line for LAMMPS (see in.lammps)
        pair_coeff=os.path.join(ref_path,potential)+" "+os.path.join(ref_path,"D.snapparam")+" X",
        
        # mass of the element
        mass = mass
        )
DDOS_Manager.run()

# Exit MPI
MPI.COMM_WORLD.Barrier()
MPI.Finalize()

exit()
