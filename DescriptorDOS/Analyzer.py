import pickle as pkl
import numpy as np 
import os
from datetime import datetime
from tqdm import tqdm
from .Constants import eV, Boltzmann, atomic_mass, hbar
from numpy.polynomial.chebyshev import Chebyshev,chebder
from numpy.polynomial.hermite import Hermite, hermder

try:
    from sklearn.linear_model import BayesianRidge
except ImportError:
    class BayesianRidge:
        def __init__(self,fit_intercept=False):
            pass
        def fit(self, X, y):
            self.coef_ = np.linalg.lstsq(X,y)[0]

class DDOSAnalyzer:
    def __init__(self,filename,\
                    volume_per_atom=None,\
                    verbose=False,mass=None):
        """
            volume_per_atom
            
        """
        
        assert not (volume_per_atom is None)

        self.kB = Boltzmann / eV # eV/K
        self.hbar = hbar / eV * 1e12 # eV ps
        self.eV_AA = eV / (atomic_mass * (1e-10)**2 / (1e-12)**2)
        

        # load pkl
        assert os.path.exists(filename)
        with open(filename,"rb") as f:
            self.raw = pkl.load(f)
        if verbose:
            print(f"""

            DDOS ANALYZER: {filename}
            {len(self.raw)}
            {self.raw[0].keys()}

              """)
        
        # Current DDOSAnalyzer only treats one volume at a time..... 
        self.volumes_per_atom = np.unique([r['V_0']/r['N'] for r in self.raw])
        self.vol_index = np.abs(volume_per_atom-self.volumes_per_atom).argmin()
        self.volume_per_atom = self.volumes_per_atom[self.vol_index]
        if verbose:
            print(f""" 
            Current DDOSAnalyzer only treats one volume at a time
            Target volume per atom: {np.round(volume_per_atom,3)}
            Available: {np.round(self.volume_per_atom,3)}
            Closest to target {volume_per_atom:.3f} is {self.volume_per_atom:.3f}%
            """)
        
        # all data points at given vol_per_atom value
        raw_match = [r for r in self.raw \
            if np.isclose(r['V_0']/r['N'],self.volume_per_atom)]
        self.flavor = raw_match[0]['flavor'].lower()
        
        self.D_0_all_vols = []
        for ss in self.volumes_per_atom:
            for r in self.raw:
                if np.isclose(r['V_0']/r['N'],ss):
                    self.D_0_all_vols += [r['D_0']]
                    break 
        self.D_0_all_vols = np.array(self.D_0_all_vols)
        if self.flavor == 'kinetic':
            self.D_0_all_vols = \
                np.hstack((self.D_0_all_vols, np.ones((self.D_0_all_vols.shape[0], 1))))
        
        # Baseline data, same for every simulation at the same vol 
        self.V_0 = raw_match[0]['V_0']
        
        if mass is None:
            self.mass = raw_match[0]['mass']
        else:
            self.mass = mass
        
        self.eval_to_hw = np.sqrt(self.eV_AA/self.mass) * self.hbar

        self.N = raw_match[0]['N']

        self.finite_size_correction = 1.0# - 1.0/self.N

        assert self.flavor in ['hessian','kinetic']
        
        # for alpha function
        self.U_0 = raw_match[0]['U_0']
        
        # For reference 
        if self.flavor == 'hessian':
            self.log_mean = raw_match[0]['log_det']
            #hbar_sq_over_mass  =  (self.hbar**2) / self.mass * self.eV_AA
            #S_0_a = np.log(3.0 * beta * hbar_sq_over_mass) + self.log_mean -1.0
            hbar_sq_over_mass  =  (self.hbar**2) / self.mass * self.eV_AA
            self.S_alpha = lambda a,T:\
                -1.5 * (np.log(3.0 * (1.0/self.kB/T) * hbar_sq_over_mass) + self.log_mean -1.0 - a)
        else:
            self.log_mean = 1.0
            self.S_alpha = None
            
        # IsoSurface data, sorted
        self.ordered_Isosurface = np.array(\
            [r['IsoSurfaceMeasured_mean'] for r in raw_match])
        order = self.ordered_Isosurface.argsort()
        sel = self.ordered_Isosurface[order]<5.0
        # lambda to get ordered array
        ordered_array = \
            lambda k: np.asarray([raw_match[i][k] for i in order[:]])
        
        self.D_0 = ordered_array('D_0')[0]

        self.isosurface_temperatures = ordered_array('IsoSurfaceMeasured_mean')
        
        self.alpha_shift = -raw_match[0]['alpha_shift']
        self.alphas = np.log(self.isosurface_temperatures) + self.alpha_shift
        
        self.E_mean = ordered_array('E_mean')
        self.T_Equipartition = ordered_array('EqiTemperature_mean')
        
        # Descriptor baseline, covariance and mean
        if 'mean' in self.raw[0]:
            self.D_mean_raw = ordered_array('mean')
            self.D_cov_raw = ordered_array('cov')
            # shape of (1,Ndesc,Ndesc)
            #self.D_per_atom_cov_raw = (self.D_cov_raw * self.N)[None,:,:]
        else:
            self.D_mean_raw = ordered_array('D_mean')
            self.D_cov_raw = ordered_array('D_var')
            # shape of (shells,Ndesc,Ndesc)
            #if 'DD_mean' in self.raw[0].keys():
            #    self.D_per_atom_cov_raw = ordered_array('DD_mean')
                # backwards compatibility
            #    if self.D_per_atom_cov_raw.ndim==2:
            #        self.D_per_atom_cov_raw = self.D_per_atom_cov_raw[None,:,:]
            #else:
            #    self.D_per_atom_cov_raw = (self.D_cov_raw * self.N)[None,:,:]
        
        if verbose:
            print(f"""
            IsoSurface: {np.round(self.isosurface_temperatures,2)}
            Energies: {np.round(self.E_mean,2)}
            Temperatures: {np.round(self.T_Equipartition,2)}
            Descriptors: {self.D_0.shape}
        """)
        
        # Rank one compression
        """ 
            D_alpha - D0 -> dD_dalpha + mean_alpha - D0
                    -> dD_rot_alpha@W_alpha + mean_alpha - D0
                     = (dD_rot_alpha + mean_rot_alpha - D0_rot_alpha)@W_alpha
            
            We have histograms of dD_rot_alpha: cov_counts,cov_bins
            We should add mean_rot_alpha to bins

            ALL DATA EXPRESSED IN BASIS OF ALPHA MODES FOR EACH ALPHA
            => ALL ARRAYS HAVE EXTRA DIMENSION OF SIZE #ALPHA (typically 20-30)
        """ 
        self.Variances = ordered_array('cov_eval')
        self.Rotations = ordered_array('cov_evec')
        
        # Mean D and D of T=0K lattice (exp(alpha)=0)
        self.D_mean = \
            np.array([mu@R for mu,R in zip(self.D_mean_raw,self.Rotations)])
        
        if self.D_0.size!=self.D_mean[0].size:
            self.D_0 = np.append(self.D_0,1.0)

        self.D_zero = \
            np.array([self.D_0@R for R in self.Rotations])
        
        # The histogram bins
        self.D_bins = ordered_array('cov_bins')
        self.D_bins = 0.5 * ( self.D_bins[:,:,1:] + self.D_bins[:,:,:-1] )
        self.D_bins += self.D_mean[:,:,None]
        
        # The histogram counts
        self.D_counts = ordered_array('cov_counts')

        self.Cumulants,self.CumulantErrors = self.calculate_cumulants()
        
        # OTF Cumulants (Experimental)
        #self.ModeCumulants = \
        #    {k:ordered_array(f"D_Cumulants_{k+1}") for k in [1,2,3]}
        #self.ModeCumulants[0] = self.D_mean.copy() - self.D_zero
        
        comp = lambda x,y: [np.abs(x-y).max(),np.abs(x).max()]
        
        # Remesh....a
        #self.D_bins = 0.5 * (self.D_bins[:,:,::2]+self.D_bins[:,:,1::2])
        #self.D_counts = 0.5 * (self.D_counts[:,:,::2]+self.D_counts[:,:,1::2])
        self.N_desc = self.D_0.size
        
        self.score_poly_order = 4
        self.match_score()
        
        # For comparisons (see below)
        self.required_keys = ['Thetas','Temperatures','F']
        self.Thetas = None
        self.test_F_H = None
        self.test_F = None
        self.test_T = None

    def load_comparison(self,EXT_DATA):
        """
            Use above definitions, only for testing
        """
        assert min([t in EXT_DATA.keys() for t in self.required_keys])
        Thetas = EXT_DATA['Thetas']
        if self.flavor == 'kinetic':
            n = self.Thetas_raw.shape[0]
            Thetas = \
                np.hstack((Thetas,np.ones_list(n).reshape(-1,1)/self.N))
        
        assert Thetas.shape[1] == self.Rotations[0].shape[0]
        self.Thetas_raw = Thetas.copy()
        self.Thetas = np.asarray([Thetas@R for R in self.Rotations])
        self.test_T = EXT_DATA['Temperatures'].copy()

        assert EXT_DATA['F'].shape[0]==self.test_T.size
        assert EXT_DATA['F'].shape[1]==self.Thetas.shape[1]

        self.test_F = EXT_DATA['F'].copy()
        
        if 'F_harmonic' in EXT_DATA.keys():
            assert EXT_DATA['F'].shape == EXT_DATA['F_harmonic'].shape
            self.test_F_H = EXT_DATA['F_harmonic'].copy()
    
    def load_prediction(self,InThetas,T,mass=None):
        """
            Only parameter values, for predictions
            We express all parameters in alpha basis (i.e. +1 dimension)
        """
        
        if self.flavor == 'kinetic':
            Thetas = np.ones((InThetas.shape[0],InThetas.shape[1]+1))
            Thetas[:,:-1] = InThetas
        else:
            Thetas = InThetas.copy()
        
        print(Thetas.shape,self.Rotations[0].shape[0])
        
        assert Thetas.shape[1] == self.Rotations[0].shape[0]
        if not mass is None:
            self.test_mass = mass 

        self.test_T = T.copy()
        self.Thetas_raw = Thetas.copy()
        
        self.E_V = self.Thetas_raw@self.D_0_all_vols.T # (nmodels,nvols)
        
        self.Thetas = np.array([Thetas@R for R in self.Rotations])
        
    def fit_S_alpha(self,Beta,F,F_error=None,n_repeats=10,U_0=None):
        if not U_0 is None:
            self.U_0 = U_0
        if self.flavor == 'hessian':
            return 
        
        assert Beta.shape==F.shape
        assert n_repeats>0

        # Feature set for beta- important for derivatives
        Features = lambda B : np.array([np.log(B)/B,     1.0/B,  1.0/B**2,  1.0/B**3]).T
        dFeatures = lambda B : np.array([(1.0-np.log(B))/B**2, -1.0/B**2, -2.0/B**3, -3.0/B**4]).T
        if F_error is None:
            A = Features(Beta)
            b = F.copy()
        else:
            # resample epistemic noise
            b = np.random.normal(0.0,1.0,(n_repeats,F.size))*0.01
            b = (b@np.diag(np.sqrt(F_error))).flatten()+np.repeat(F,n_repeats)
            A = np.repeat(Features(Beta),n_repeats,axis=0)
        
        # Fitting
        m = BayesianRidge(fit_intercept=False)
        m.fit(A,b)
        w = m.coef_.copy()
        
        # Remeshing
        B_A = 1.0/np.linspace(1.0/Beta[-1],1.0/Beta[0],3*Beta.size)
        F_A = Features(B_A)
        U_A = Features(B_A) + B_A[:,None]*dFeatures(B_A)

        # entropy and alpha
        S_A = B_A*((U_A-F_A)@w)
        a_A = np.log(U_A@w/self.U_0)
        
        # Fitting cubic for S_A
        Features = lambda a:np.array([np.ones_like(a),a,a**2]).T
        m = BayesianRidge(fit_intercept=False)
        m.fit(Features(a_A),S_A)
        self.S_alpha_coef = m.coef_
        # DDOS_S -= 1.0 * beta  
        self.S_alpha = lambda a,T=0.0 : Features(a)@self.S_alpha_coef - 1.0/(self.kB*T)


    def calculate_cumulants(self,filter_width=1.0):
        from scipy.integrate import simpson
        from scipy.ndimage import gaussian_filter1d
        Cumulants = np.zeros((4,*self.Variances.shape))
        CumulantErrors = np.zeros((4,*self.Variances.shape))
        
        CA = self.D_counts.copy() # (nalpha,ndesc,nbins)
        NH = self.D_counts.sum(-1) # (nalpha,ndesc)
        rho = np.array([np.array([\
                        gaussian_filter1d(c,\
                            filter_width*np.abs(c[1:]-c[:-1]).mean(0)\
                        ) 
                    for c in C]) for C in CA])

        # mean
        D = self.D_bins.copy() - self.D_zero[:,:,None]

        rho = 1.0*rho / simpson(rho,x=D,axis=-1)[:,:,None]
        Cumulants[0] = simpson(rho * D,x=D,axis=-1)
        
        # variance
        cD = D.copy() - Cumulants[0][:,:,None]
        Cumulants[1] = self.N * simpson(rho * cD**2,x=D,axis=-1)
        
        # skewness
        Cumulants[2] = self.N**2 * simpson(rho * cD**3,x=D,axis=-1)
        
        # kurtosis / 4th cumulant
        Cumulants[3] = simpson(rho * cD**4,x=D,axis=-1)
        Cumulants[3] -= 3.0 * ( simpson(rho * cD**2,x=D,axis=-1) )**2
        Cumulants[3] *= self.N**3

        # Errors
        CumulantErrors[0] = np.sqrt(Cumulants[1])/np.sqrt(NH)
        CumulantErrors[1] = Cumulants[1]/np.sqrt(NH) * np.sqrt(2.0)
        CumulantErrors[2] = Cumulants[1]*np.sqrt(Cumulants[1])/np.sqrt(NH) * np.sqrt(6.0)
        CumulantErrors[3] = Cumulants[1]**2/np.sqrt(NH) * np.sqrt(24.0)
        


        

        return Cumulants.copy(),CumulantErrors.copy()
    
    def cohesive_energy(self,Thetas=None):
        if Thetas is None:
            return self.Thetas_raw@self.D_0
        else:
            return Thetas@self.D_0
        
    def cumulant_free_energy(self,Thetas=None,T=1000,order=0,fourth=False,return_D=False,ext_factor=1.0):
        """
            This function does the full prediction for each Theta

            order: polynomial order for interpolation. If order==0, we spline. 

            first, find each max at fixed alpha, then maximise wrt alpha
        """
        assert order>=0
        # Thetas: unrotated coefficients
        # ThetasAlpha: rotated coefficients
        if not Thetas is None:
            ThetasAlpha = np.array([Thetas@R for R in self.Rotations])
        else:
            ThetasAlpha = self.Thetas.copy() # poor notation here!
            Thetas = self.Thetas_raw.copy()
        
        start_time = datetime.now()
        
        # Reference free energy as function of alpha
        # We isolate S_0(alpha)
        assert not self.S_alpha is None

        lnZ_0 = self.S_alpha(self.alphas,T)
        
        lnZ_alpha_all = []
        for C in [self.Cumulants,\
                self.Cumulants+self.CumulantErrors/2.0,\
                self.Cumulants-self.CumulantErrors/2.0]:
            
            lnZ_alpha = np.outer(lnZ_0,np.ones(Thetas.shape[0]))
            betaT = 1.0/self.kB/T * ThetasAlpha # (nalpha,nmodels,ndesc)
            lnZ_alpha -= np.einsum('ijk,ik->ij',betaT,C[0])
            lnZ_alpha += np.einsum('ijk,ik->ij',betaT**2,C[1])/2.0
            if fourth:
                lnZ_alpha -= np.einsum('ijk,ik->ij',betaT**3,C[2])/6.0
                lnZ_alpha += np.einsum('ijk,ik->ij',betaT**4,C[3])/24.
            lnZ_alpha_all += [lnZ_alpha]
        
        lnZ_alpha = np.array(lnZ_alpha_all)    
        
        # alpha-dependent free energy - note correction to be *per-mode*
        F_alpha = (-lnZ_alpha[0]*self.kB*T)*self.finite_size_correction
        F_alpha_u = (-lnZ_alpha[1]*self.kB*T)*self.finite_size_correction
        F_alpha_l = (-lnZ_alpha[2]*self.kB*T)*self.finite_size_correction
        
        dF_alpha = np.gradient(F_alpha,self.alphas,axis=0)
        ddF_alpha = np.gradient(dF_alpha,self.alphas,axis=0)
        
        F = F_alpha.min(0)
        eF = np.abs(F_alpha_u.min(0)-F)+np.abs(F_alpha_l.min(0)-F)

        ialpha_min = F_alpha.argmin(0)
        alpha_star = self.alphas[ialpha_min]
        
        unsolved_models = F_alpha.argmin(0)==F_alpha.shape[0]-1
        unsolved_models += F_alpha.argmin(0)==0
        if order==0:
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
        
            res = {\
                'F':F,\
                'errorF':eF,\
                'alpha_star':alpha_star,\
                'unsolved_models':unsolved_models,\
                'F_alpha':F_alpha.copy(),\
                'alphas' : self.alphas,
                'elapsed' : elapsed}
            if return_D:
                res['D'] = self.D_mean_raw[ialpha_min].copy()
            return res


        # fit a polynomial... 
        poly_feat_0 = lambda a:np.array([a**n for n in range(order+1)]).T
        poly_alpha_0 = poly_feat_0(self.alphas) # nalpha,ncoeffs
        
        coefs = []
        for i in range(F_alpha.shape[1]):
            convex = ddF_alpha[:,i]>=0.0
            #convex[convex.argmin()-1:]=False
            #convex += True
            coefs +=[\
                np.linalg.lstsq(poly_alpha_0[convex],F_alpha[convex,i])[0]]
        
        coefs = np.array(coefs)
        
        dense_alphas = np.linspace( self.alphas[0]-np.log(ext_factor),\
                                    self.alphas[-1]+np.log(ext_factor),\
                                    10*self.alphas.size)
        
        dense_F_alpha = poly_feat_0(dense_alphas)@coefs.T
        
        
        ia = dense_F_alpha.argmin(0)
        alpha_star = dense_alphas[ia]

        alpha_lower_index = np.argmin(np.abs(self.alphas[:,None] - alpha_star[None,:]),axis=0)
        if Thetas.shape[0]==1:
            alpha_lower_index = alpha_lower_index.reshape((-1,))
        
        alpha_upper_index = np.where(alpha_lower_index<self.alphas.size-1,alpha_lower_index + 1,alpha_lower_index)
        diff_alpha = alpha_star - self.alphas[alpha_lower_index]
        diff_alpha /= np.where(alpha_lower_index<self.alphas.size-1,self.alphas[alpha_upper_index] - self.alphas[alpha_lower_index],1.0)
        
        if return_D:
            dense_D = self.D_mean_raw[alpha_lower_index].copy()
            dense_D += diff_alpha * (self.D_mean_raw[alpha_upper_index]-\
                                     self.D_mean_raw[alpha_lower_index])
        else:
            dense_D = None
        
        dense_F = dense_F_alpha[ia,np.arange(ia.size)]
        
        unsolved_models_dense = ia==dense_alphas.size-1
        unsolved_models_dense += ia==0
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        return {\
                'F':dense_F,\
                'errorF':eF,\
                'F_raw' : F,\
                'alpha_star':alpha_star,\
                'alphas_raw' : self.alphas,
                'unsolved_models_raw':unsolved_models,\
                'F_alpha_raw':F_alpha.copy(),\
                'D_raw' : self.D_mean_raw[ialpha_min].copy(),\
                'elapsed' : elapsed,\
                'D':dense_D,\
                'alphas' : dense_alphas,
                'F_alpha':dense_F_alpha.copy(),
                'unsolved_models':unsolved_models_dense}
    
    def poly_features(self,x,type='monomial',order=4,der=0):
        assert type.lower() in ['monomial','chebyshev','hermite']
        assert order>=0
        assert der>=0 and der<=2 
        
        poly_features = None 
        d_poly_features = None
        dd_poly_features = None
        
        if type.lower() == 'chebyshev':
            poly_indices = (np.eye(order)+1)[1:]
            poly_features = np.array([Chebyshev(p)(x) for p in poly_indices])
            if der>0:
                d_poly_indices = [chebder(p) for p in poly_indices]
                d_poly_features = np.array([Chebyshev(p)(x) for p in d_poly_indices])
            if der>1:
                dd_poly_indices = [chebder(p) for p in d_poly_indices]
                dd_poly_features = np.array([Chebyshev(p)(x) for p in dd_poly_indices])
        elif type.lower() == 'hermite':
            poly_indices = (np.eye(order)+1)[1:]
            poly_features = np.array([Hermite(p)(x) for p in poly_indices])
            if der>0:
                d_poly_indices = [hermder(p) for p in poly_indices]
                d_poly_features = np.array([Hermite(p)(x) for p in d_poly_indices])
            if der>1:
                dd_poly_indices = [hermder(p) for p in d_poly_indices]
                dd_poly_features = np.array([Hermite(p)(x) for p in dd_poly_indices])
        else:
            poly_features = np.array([x**(i+1) for i in range(order)])
            if der>0:
                d_poly_features = np.array([(i+1) * x**i for i in range(order)])
            if der>1:
                dd_poly_features = np.array([i * (i+1) * x**(i-1) if i>0 else 0.0*x for i in range(order)])
        if der==2:
            return poly_features,d_poly_features,dd_poly_features
        if der==1:
            return poly_features,d_poly_features
        return poly_features

    def match_score(self,poly_order=4,poly_type='monomial'):
        """
            Fit a polynomial score model
        """
        # The histogram counts
        CA = self.D_counts.copy() # (nalpha,ndesc,nbins)
        # Total number of counts per descriptor, alpha
        NH = self.D_counts.sum(-1) # (nalpha,ndesc)
        # Descriptor argument
        D = self.D_bins.copy() - self.D_zero[:,:,None] # (nalpha,ndesc,nbins)
        # Gaussian normalization
        dD = (D - D.mean(-1)[:,:,None])/D.std(-1)[:,:,None] # (nalpha,ndesc,nbins)

        poly_features,d_poly_features,dd_poly_features = \
            self.poly_features(dD,poly_type,poly_order,2)

        # Density
        rho = 1.0 * CA / CA.sum(-1)[:,:,None]
        # Feature matching
        M = np.einsum('iabc,jabc->ijabc',d_poly_features,d_poly_features)
        
        # We therefore have a matrix A and a vector b
        A = (M*rho[None,None,:,:,:]).sum(-1) * self.N
        b = (dd_poly_features*rho[None,:,:,:]).sum(-1)

        MTM = np.einsum('ikabc,kjabc->ijabc',M,M)
        A_var = (MTM*rho[None,None,:,:,:]).sum(-1)
        A_var -= np.einsum('ikab,kjab->ijab',A,A)
        A_var /= NH[None,None,:,:] / 3.0
        A_var *= self.N**2
        
        M = np.einsum('iabc,jabc->ijabc',dd_poly_features,dd_poly_features)
        b_var = (M*rho[None,:,:,:]).sum(-1)
        b_var -= np.einsum('iab,jab->ijab',b,b)
        b_var /= NH[None,None,:,:] / 3.0 
        
        coefs = np.zeros(b.shape)
        coefs_var = np.zeros(A.shape)
        for i in range(A.shape[-2]):
            for j in range(A.shape[-1]):
                # Solve for coefficients
                x = -np.linalg.inv(np.eye(b.shape[0])/self.N+A[:,:,i,j])@b[:,i,j]
                #np.linalg.solve(A[:,:,i,j],-b[:,i,j])
                # next-order has no impact
                #dx = 1.0*np.linalg.lstsq(self.N**2 * A[:,:,i,j],-self.N**2 * (b[:,i,j]+A[:,:,i,j]@x))[0]
                coefs[:,i,j] = x.copy()
                
                # inv(A).x
                iAx = np.linalg.solve(A[:,:,i,j],x)
                V = np.linalg.solve(A[:,:,i,j],b_var[:,:,i,j].T).T
                V += A_var[:,:,i,j]@np.outer(iAx,iAx)
                coefs_var[:,:,i,j] = np.linalg.solve(A[:,:,i,j],V)
        
        S = (coefs[:,:,:,None] * poly_features).sum(0)


        S_var = np.einsum('ijab,iabc,jabc->abc',coefs_var,poly_features,poly_features)
        Se = np.sqrt(np.abs(S_var)*1.0)
        S -= S.max(-1)[:,:,None]
        self.S = S
        self.Su = S+Se
        #self.Su -= self.Su.max(-1)[:,:,None]
        self.Sl = S-Se
        #self.Sl -= self.Sl.max(-1)[:,:,None]
    
    
    def score_free_energy(self,Temperature,\
                        Thetas=None,\
                        alpha_poly_order=5,\
                        alpha_poly_type='monomial',\
                        alpha_remesh=2,ext_factor=1.0,
                        return_D=False):
        
        # Thetas: unrotated coefficients
        # ThetasAlpha: rotated coefficients
        if not Thetas is None:
            ThetasAlpha = np.array([Thetas@R for R in self.Rotations])
        else:
            ThetasAlpha = self.Thetas.copy() # poor notation here!
            Thetas = self.Thetas_raw.copy()

        start_time = datetime.now()
        assert alpha_poly_order>=0
        assert alpha_poly_type.lower() in ['monomial','chebyshev','hermite']

        D = self.D_bins.copy() - self.D_zero[:,:,None]
        BwD = ThetasAlpha[:,:,:,None] * D[:,None,:,:] / (self.kB*Temperature)
        
        BF = BwD - self.S[:,None,:,:] # nalphas,nmodels,ndesc,nbins
        BF = BF.min(-1).sum(-1) # nalphas,nmodels

        # entropy bounds
        BFu = BwD - self.Su[:,None,:,:]
        BFu = BFu.min(-1).sum(-1) # nalphas,nmodels

        BFl = BwD - self.Sl[:,None,:,:]
        BFl = BFl.min(-1).sum(-1) # nalphas,nmodels
        
        # no interpolation
        BF_0 = -self.S_alpha(self.alphas,Temperature)
        BF_raw = (BF.copy() + BF_0[:,None])
        ia_raw = BF_raw.argmin(0)
        alpha_star_raw = self.alphas[ia_raw]
        

        F_raw = (BF_raw.min(0) * (self.kB*Temperature))*self.finite_size_correction
        solved_raw = (BF_raw.argmin(0)>0) * (BF_raw.argmin(0)<BF_raw.shape[0]-1)
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        # alpha poly features: n_feat,nalpha
        a_p_f = np.array([self.alphas**(i) for i in range(alpha_poly_order+1)])
        coefs = np.linalg.solve(a_p_f@a_p_f.T,a_p_f@BF)
        coefs0 = np.linalg.solve(a_p_f@a_p_f.T,a_p_f@BF_0)
        coefsu = np.linalg.solve(a_p_f@a_p_f.T,a_p_f@BFl)
        coefsl = np.linalg.solve(a_p_f@a_p_f.T,a_p_f@BFu)

        dense_alphas = np.linspace( self.alphas[0]-np.log(ext_factor),\
                                        self.alphas[-1]+np.log(ext_factor),\
                                        alpha_remesh*self.alphas.size)
        BF = np.array([dense_alphas**(i) for i in range(alpha_poly_order+1)]).T@coefs
        BF0 = np.array([dense_alphas**(i) for i in range(alpha_poly_order+1)]).T@coefs0
        BFl = np.array([dense_alphas**(i) for i in range(alpha_poly_order+1)]).T@coefsl
        BFu = np.array([dense_alphas**(i) for i in range(alpha_poly_order+1)]).T@coefsu
        
        BFe = np.abs(BFu-BF)+np.abs(BFl-BF)
        
        scale = self.finite_size_correction * self.kB * Temperature
        F = (BF+BF0[:,None])*scale
        Fu = (BFu+BF0[:,None])*scale
        Fl = (BFl+BF0[:,None])*scale
        Fe = BFe * scale



        # return 

        ia = F.argmin(0)
        alpha_star = dense_alphas[ia]

        alpha_lower_index = np.argmin(np.abs(self.alphas[:,None] - alpha_star[None,:]),axis=0)
        if Thetas.shape[0]==1:
            alpha_lower_index = alpha_lower_index.reshape((-1,))
        
        alpha_upper_index = np.where(alpha_lower_index<self.alphas.size-1,alpha_lower_index + 1,alpha_lower_index)
        diff_alpha = alpha_star - self.alphas[alpha_lower_index]
        diff_alpha /= np.where(alpha_lower_index<self.alphas.size-1,self.alphas[alpha_upper_index] - self.alphas[alpha_lower_index],1.0)
        
        if return_D:
            dense_D = self.D_mean_raw[alpha_lower_index].copy()
            dense_D += diff_alpha * (self.D_mean_raw[alpha_upper_index]-\
                                     self.D_mean_raw[alpha_lower_index])
        else:
            dense_D = None
        
        
        solved = (F.argmin(0)>1) * (F.argmin(0)<F.shape[0]-1)

        return {'F_raw':F_raw,\
                'unsolved_models_raw':~solved_raw,\
                'alpha_star_raw':alpha_star_raw,\
                'F':F.min(0),\
                'unsolved_models':~solved,\
                'errorF':0.5*(np.abs(Fl.min(0)-F.min(0))+np.abs(Fu.min(0)-F.min(0))),\
                'elapsed' : elapsed,\
                'F_alpha' : F,
                'Fe_alpha' : Fe,
                'Fl_alpha' : Fu,
                'Fu_alpha' : Fl,
                'D':dense_D,\
                'alpha':dense_alphas,
                'alpha_star':dense_alphas[F.argmin(0)]}

