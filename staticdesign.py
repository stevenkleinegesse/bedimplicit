#!/usr/bin/env python3

import numpy as np

# own libraries
import inference
import utility
import methods

# for Bayesian Optimization
from GPyOpt.methods import BayesianOptimization


class StaticBED:

    """
    Class that performs static experimental design for a given simulator object and prior samples.
    """

    def __init__(self, prior_samples, simobj, domain, constraints=None, num_cores=1, utiltype='MI'):

        """
        prior_samples: samples from the prior distribution
        simobj: simulator object of the simulator model
        domain: GPyOpt domain of design variable
        constraints: GPyOpt constraints of design variable
        """
        self.prior_samples = prior_samples
        self.simobj = simobj
        self.domain = domain
        self.constraints = constraints
        self.num_cores = num_cores

        # choose the utility used in the optimisation
        self.utiltype = utiltype
        if self.utiltype=='MI':
            # set uniform weights
            self.weights = np.ones(len(self.prior_samples))

            # define the utility object
            self.utilobj = utility.MutualInformation(self.prior_samples, self.weights, self.simobj, evalmethod='lfire')
        else:
            raise NotImplementedError()

    def _objective(self, d):

        u = - self.utilobj.compute(d, numsamp=10*len(self.prior_samples), evaltype='robust')
        return u

    def optimisation(self, init_num=5, max_iter=10):

        # Define GPyOpt Bayesian Optimization object
        myBopt = BayesianOptimization(f=self._objective, domain=self.domain, constraints=self.constraints, acquisition_type='EI', normalize_Y=True, initial_design_numdata=init_num, evaluator_type='local_penalization', batch_size=int(self.num_cores), num_cores=int(self.num_cores), acquisition_jitter=0.01)

        # run the bayesian optimisation
        myBopt.run_optimization(max_iter=max_iter)
        self.bo_obj = myBopt

        # Select method to get optimum
        #optmethod='point' # take optimum from BO evaluations
        optmethod='interpol' # use posterior predictive to find optimum
        if optmethod=='point':
            d_opt = self.bo_obj.x_opt
        elif optmethod=='interpol':
            d_opt = methods.get_GP_optimum(self.bo_obj)
        else:
            raise NotImplementedError()

        # Take some real-world data at optimum
        y_obs = self.simobj.observe(d_opt)[0]

        if self.utiltype=='MI':

            # Compute ratios r_obs and coefficients b_obs for final observation
            r_obs, b_obs = self.utilobj.compute_final(d_opt, y_obs, num_cores=self.num_cores)

            self.savedata = {'d_opt':d_opt, 'y_obs':y_obs, 'r_obs':r_obs, 'b_obs':b_obs}

        else:
            raise NotImplementedError()

    def save(self, filename):

        np.savez('{}.npz'.format(filename), **self.savedata, utilobj=self.utilobj, bo_obj=self.bo_obj, prior_samples=self.prior_samples)
