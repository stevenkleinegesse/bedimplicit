#!/usr/bin/env python3

import numpy as np
import inference
import robust

# for parallel processing
from joblib import Parallel, delayed
import multiprocessing

class Utility:

    """
    Base class to implement a utility function.
    """

    def __init__(self, prior_samples, weights, simobj):

        """
        prior_samples: samples from the prior distribution
        weights: a weight for each prior sample
        simobj: simulator object
        """

        self.prior_samples = prior_samples
        self.weights = weights
        self.simobj = simobj

class MutualInformation(Utility):


    def __init__(self, prior_samples, weights, simobj, evalmethod='lfire'):

        """
        prior_samples: samples from the prior distribution
        weights: a weight for each prior sample
        simobj: simulator object
        type: type of mutual information implementation
        """

        super(MutualInformation, self).__init__(prior_samples, weights, simobj)
        self.evalmethod = evalmethod

    def _mean_eval(self, U):

        """
        U: array of utilities
        """

        return np.mean(U)

    def _median_eval(self, U):

        """
        U: array of utilities
        """

        return np.median(U)

    def _robust_eval(self, U):

        """
        U: array of utilities
        """

        return robust.MEstimator(U)


    def compute(self, d, numsamp=10000, evaltype='robust', verbose=True):

        """
        Compute mutual information given the evaluation method (default is 'lfire').

        d: design variable
        evaltype: type of Monte-Carlo evaluation; 'mean', 'median' or 'robust'
        """

        # GPyOpt wraps the design point in a weird double array // hacky fix
        d = d[0]

        if verbose:
            print('Design point: ', d)

        if self.evalmethod=='lfire':

            # Define LFIRE object
            infobj = inference.LFIRE(d, self.prior_samples, self.weights, self.simobj)

            # compute the LFIRE ratios for 'numsamp' prior samples, where some may be repeated; set to default 10000 for now.
            utils, _ = infobj.ratios(numsamp=numsamp)

            self.utils = np.array(utils)
            # self.coefs = coefs
        else:
            raise NotImplementedError()

        if evaltype=='mean':
            mutualinfo = self._mean_eval(self.utils)
        elif evaltype=='median':
            mutualinfo = self._median_eval(self.utils)
        elif evaltype=='robust':
            mutualinfo = self._robust_eval(self.utils)
        else:
            raise NotImplementedError()

        return mutualinfo

    def compute_final(self, d_opt, y_obs, num_cores=1):

        """
        Final likelihood-free inference for the optimal design.

        d_opt: optimal design

        """

        if self.evalmethod=='lfire':

            # Define LFIRE object
            infobj = inference.LFIRE(d_opt, self.prior_samples, self.weights, self.simobj)

            # Take summary statistics of observed data
            if len(y_obs.shape) > 2:
                psi_obs = self.simobj.summary(y_obs)
            else:
                psi_obs = self.simobj.summary(y_obs.reshape(1, -1))

            # Compute coefficients for each prior sample
            tmp_bl = Parallel(n_jobs=int(num_cores))(delayed(infobj._logistic_regression)(p) for p in self.prior_samples)
            self.b_obs = np.array(tmp_bl)

            # Compute ratios for each coefficient
            self.r_obs = np.array([np.exp(psi_obs.reshape(1, -1) @ b[1:] + b[0])[0][0] for b in self.b_obs])

            return self.r_obs, self.b_obs

        else:
            raise NotImplementedError()

