#!/usr/bin/env python3

import numpy as np
from math import isinf
import glmnet

class Inference:

    """
    Base class for Ratio Estimation Inference for a particular simulator model.
    """

    def __init__(self, simobj):

        """
        simobj: simulator object of the implicit simulator model
        """

        self.simobj = simobj

    def ratios(self):
        pass


class LFIRE(Inference):

    """
    Class that implements Likelihood-Free Inference by Ratio Estimation (LFIRE) for a simulator models. The ratios() method returns a set of log ratios, as well as logistic regression coefficients corresponding to prior samples.
    """

    def __init__(self, d, prior_samples, weights, simobj, psi=None):

        """
        d: design variable
        prior_samples: samples from the prior distribution; of shape (, dim)
        weights: weights for each prior sample
        simobj: simulator object of the implicit simulator model
        psi: summary statistics function; use 'simobj.summary' if unsure
        """

        super(LFIRE, self).__init__(simobj)
        self.d = d
        self.prior_samples = prior_samples
        self.weights = weights
        self.simobj = simobj
        # summary statistics
        if psi == None:
            self.psi = self.simobj.summary
        else:
            if callable(psi):
                self.psi = psi
            else:
                raise TypeError('Your summary statistics needs to be a function.')

    def _logistic_regression(self, theta, K=10):

        """
        Returns logistic regression coefficients.

        theta: model parameter for which to compute coefficients
        K: folds of cross-validation
        """

        # Select model params according to weights
        ws_norm = self.weights / np.sum(self.weights)
        p_selec = list()
        idx_selec = list()
        for _ in range(self.prior_samples.shape[0]):
            cat = np.random.choice(range(len(ws_norm)), p=ws_norm)
            p_selec.append(self.prior_samples[cat])
            idx_selec.append(cat)

        # Simulate from marginal using selected model params
        y_m = self.simobj.sample_data(self.d, p_selec)
        y_m = self.simobj.summary(y_m)

        # Simulate from likelihood
        y_t = self.simobj.sample_data(self.d, theta, num=len(self.prior_samples))
        y_t = self.simobj.summary(y_t)

        # Prepare targets
        t_t = np.ones(y_t.shape[0])
        t_m = np.zeros(y_m.shape[0])

        # Concatenate data
        Y = np.concatenate((y_t, y_m), axis=0)
        T = np.concatenate((t_t, t_m))

        # Define glmnet model
        model = glmnet.LogitNet(n_splits=K, verbose=False, n_jobs=1, scoring='log_loss')
        model.fit(Y, T)

        # collect coefficients and intercept
        coef_choice=model.coef_path_[..., model.lambda_max_inx_].T.reshape(-1)
        inter = model.intercept_path_[..., model.lambda_max_inx_]
        coef = np.array(list(inter) + list(coef_choice)).reshape(-1, 1)

        return coef

    def ratios(self, numsamp=10000):

        """
        Returns a set of LFIRE ratios of size 'numsamp', as well as coefficients for each model parameter.

        numsamp: number of LFIRE ratios you want. Shouldn't be much bigger than the number of prior samples.
        verbose: if 'True', print out the design that you are computing the ratios at.
        """

        # Normalise weights
        ws_norm = self.weights / np.sum(self.weights)

        # Prepare lookup dictionary such that we need to only compute ratios for each parameter sample once
        lookup = dict()
        for i in range(self.prior_samples.shape[0]):
            lookup[i] = np.array([0])

        # For each sample from the prior distribution, compute ratios
        self.kld = list()
        for _ in range(numsamp):

            # sample data from selected prior sample
            cat = np.random.choice(range(len(ws_norm)), p=ws_norm)
            y = self.simobj.sample_data(self.d, self.prior_samples[cat], num=1)

            # compute summary statistics of data
            if len(y.shape) > 2:
                psi_y = self.simobj.summary(y)
            else:
                psi_y = self.simobj.summary(y.reshape(1, -1))

            # check if we haven't already computed logistic regression coefficients for that particular prior sample
            if (lookup[cat] == 0).all():
                b = self._logistic_regression(self.prior_samples[cat])
                lookup[cat] = b
            else:
                b = lookup[cat]

            # compute the log ratio of the data y with the coefficients
            logr = psi_y.reshape(1, -1) @ b[1:] + b[0]
            logr = logr[0][0]

            # store the log ratio
            if not isinf(logr):
                self.kld.append(logr)

        # Store the coefficients for each parameter sample in a list
        self.betas = list(lookup.values())

        #self.kld = np.array(self.kld)
        #self.betas = np.array(self.betas)

        #print(self.kld.shape, self.betas.shape)

        return self.kld, self.betas
