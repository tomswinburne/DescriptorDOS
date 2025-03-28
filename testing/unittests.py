import unittest,os,shutil
SELF_DIR = os.path.dirname(os.path.abspath(__file__))

class DescriptorDOSTestCase(unittest.TestCase):
    def test1_libraries(self):
        """
            Test case for importing LAMMPS package
        """
        import numpy
        from mpi4py import MPI
        from lammps import lammps
        self.assertTrue(MPI)
        self.assertTrue(lammps)
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.assertTrue(rank>=0)
        self.assertTrue(size>0)

    
    def test2_load_manager(self):
        from mpi4py import MPI
        from DescriptorDOS import DDOSManager 
        DDOS_Manager = DDOSManager( comm=MPI.COMM_WORLD,
                                    yaml_file='test.yaml')
        DDOS_Manager.close()
        self.assertTrue(True)

    def test3_run_manager(self):
        from mpi4py import MPI
        from DescriptorDOS import DDOSManager 
        DDOS_Manager = DDOSManager( comm=MPI.COMM_WORLD,
                                    yaml_file='test.yaml')
        
        DDOS_Manager.run()
        DDOS_Manager.close()
        self.assertTrue(True)
    
    def test4_kinetic(self):
        from mpi4py import MPI
        from DescriptorDOS import DDOSManager 
        DDOS_Manager = DDOSManager( comm=MPI.COMM_WORLD,
                                    yaml_file='test.yaml',
                                    isosurface="Kinetic")
        DDOS_Manager.run()
        DDOS_Manager.close()
        self.assertTrue(True)
    
    def test5_hessian(self):
        from mpi4py import MPI
        from DescriptorDOS import DDOSManager 
        DDOS_Manager = DDOSManager( comm=MPI.COMM_WORLD,
                                    yaml_file='test.yaml',
                                    isosurface="Hessian")
        DDOS_Manager.run()
        DDOS_Manager.close()
        self.assertTrue(True)
    
    def test6_analyser(self):
        from mpi4py import MPI
        from DescriptorDOS import DDOSAnalyzer 
        DDOS_Manager = DDOSManager( comm=MPI.COMM_WORLD,
                                    yaml_file='test.yaml',
                                    isosurface="Hessian")
        DDOS_Manager.run()
        DDOS_Manager.close()
        self.assertTrue(True)



        
if __name__ == '__main__':
    output_dir = os.path.join(SELF_DIR, 'output')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    unittest.main()
