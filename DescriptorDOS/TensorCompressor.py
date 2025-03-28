import numpy as np
import time,os

from mpi4py import MPI

def TensorCompressor(UncompressedData,N=1,CompressedData=None,AuxillaryField=None,bins=100,bootstrap_ensemble_size=10,threshold_percentile=1.0):
    """
		Perform rank-one compression on the given data.
		This method computes the covariance matrix of the input data, performs
		eigenvalue decomposition, and then projects data onto the eigenvectors. 
		It then computes histograms for each projected dimension.
		
		If CompressedData is provided, it updates the existing compression results
		with the new data, allowing for incremental updates to the statistics.

		Parameters:
		-----------
		UncompressedData : numpy.ndarray
			The input data matrix where each row is a data point and each column
			is a feature.
        
        N: int, optional
            Each data point is taken as an average over N i.i.d. samples, default = 1
        
        CompressedData : dict, optional
			Results from a previous compression. If None, performs first-time compression.
        
        AuxillaryField: 
            Concatenated to descriptor

        bins: int, optional
            Number of bins for histogram, default = 100
        
        threshold_percentile: float, optional
            Percentile threshold for histogram bins, default = 5
		
		Returns:
		--------
		res : dict
			A dictionary containing the following keys:
			- 'calls' : number of data points used in compression
			- 'D_mean': The mean of the input data.
			- 'D_cov': The covariance matrix of the input data.

			- 'cov_eval': The eigenvalues of the covariance matrix.
			- 'cov_evec': The eigenvectors of the covariance matrix.
			- 'cov_counts': The histogram counts for each projected dimension.
			- 'cov_bins': The histogram bin edges for each projected dimension.
	"""
    
    if AuxillaryField is None:
        Data = UncompressedData
    else:
        n_samples = UncompressedData.shape[0]
        assert AuxillaryField.shape[0] == n_samples
        Data = np.hstack((UncompressedData,AuxillaryField.reshape((n_samples,-1))))
    
    if CompressedData is None:

        """ First time compression """
        # Compute the covariance matrix via bootstrap
        assert bootstrap_ensemble_size > 0 and isinstance(bootstrap_ensemble_size,int)
        bootstrap_cov = np.zeros((bootstrap_ensemble_size,Data.shape[1],Data.shape[1]))
        if bootstrap_ensemble_size==1:
            bootstrap_cov[0] = np.cov(Data.T)
        else:
            # require at least 10 samples per dimension 
            bag_size = max(Data.shape[1]*10,Data.shape[0]//bootstrap_ensemble_size)
            for e in range(bootstrap_ensemble_size):
                bootstrap_cov[e] = np.cov(Data[np.random.choice(Data.shape[0],bag_size)].T)
        
        cov = np.cov(Data.T)#bootstrap_cov.mean(0)
        cov_err = bootstrap_cov.std(0) * np.sqrt(bootstrap_ensemble_size)
        del bootstrap_cov
        nu,W = np.linalg.eigh(cov)
        
        nu_err,W_err = np.linalg.eigh(cov+cov_err)
        
        
        mu = Data.mean(0)
        D_t = (Data-mu[np.newaxis,:])@W # should be ~independent columns
        hist_counts = []
        hist_bins = []
        
        try:
            p_thresh = min(np.abs(threshold_percentile),30.0)
        except:
            p_thresh = 0.0
        
        for d in D_t.T:
            low_thresh = np.percentile(d,p_thresh)
            high_thresh = np.percentile(d,100. - p_thresh)
            cb = np.histogram(d,bins=np.linspace(low_thresh,high_thresh,bins))
            hist_counts += [cb[0]]
            hist_bins += [cb[1]]
        res = {'cov_eval':nu, 'cov_evec':W,'cov_evec_err':W_err,'cov_eval_err':nu_err}
        res['cov_counts'] = np.array(hist_counts)
        res['cov_bins'] = np.array(hist_bins)
        res["D_mean"] = mu.copy()
        res["D_var"] = cov.copy()
        
    else:
        """
            Subsequent compression 
            We use original mapping to project new data
            So only cov_counts is updated
        """
        mu = CompressedData['D_mean']
        counts = CompressedData['cov_counts'].copy()
        W = CompressedData['cov_evec']
        D_t = (Data-mu[np.newaxis,:])@W # should be ~independent columns
        
        # Append new data to existing counts
        for i, d in enumerate(D_t.T):
            new_counts, _ = np.histogram(d, bins=CompressedData['cov_bins'][i])
            counts[i] += new_counts
        
        # Update the results dictionary
        res = CompressedData.copy()
        res['cov_counts'] = counts.copy()
        
    return res