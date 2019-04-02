#!/usr/bin/env python3

import numpy as np
import warnings

class Simulator:

    """
    Simulator base class for simulating data from different models.
    """

    def __init__(self, truth):
        """
        truth: Ground truth that is used in the observe() method. Needs to be same dimensions and shape as the model parameters.
        """
        self.truth = np.array(truth)

    def summary(self, Y):

        """
        Method to take summary statistics of the simulated data. Default is simply powers from 1 to 4 of the data values; this is only applicable to scalars.
        Y: Data simulated from the model.
        """

        Y_psi = list()
        for y in Y:
            # could change 5 to any kind of degree
            Y_psi.append([y ** i for i in range(1, 5)])
        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        Child class specific method to simulate data from the model.
        d: design variable
        p: model parameters
        """

        pass

    def sample_data(self, d, p, num=None):

        """
        Sample data from the model, based on the generate_data() method. The point of this method is to select if to sample from the likelihood (if num!=None and len(p)==1) or marginal (if num==None and len(p) > 1).
        d: design variable
        p: model parameters
        num: number of samples required from likelihood
        """

        # sample from an array of params
        if num==None:
            y = np.array([self.generate_data(d, pi) for pi in p])
        # sample several times using the same params:
        else:
            y = np.array([self.generate_data(d, p) for i in range(num)])
        return y

    def observe(self, d, num=1):

        """
        Observe some data according to a ground truth.
        d: design variable (optimal)
        num: number of data points to observe at optimal design
        """

        y = np.array([self.generate_data(d, self.truth) for i in range(num)])
        return y


class DeathModel(Simulator):

    """
    Class to simulate data according to the Death Model.
    """

    def __init__(self, truth, S0):

        """
        truth: ground truth, scalar
        S0: starting population of death model
        """

        super(DeathModel, self).__init__(truth)
        self.S0 = S0

    def summary(self, Y):

        Y_psi = list()
        for arr in Y:
            if np.array(arr).shape == ():
                tmp = [arr, 0]
                #tmp = [arr ** i for i in range(1,5)]
            else:
                tmp = [arr[0], 0]
                #tmp = [arr[0] ** i for i in range(1, 5)]
            Y_psi.append(tmp)
        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        d: design (scalar)
        p: model parameter (scalar)
        """

        inf_prob = lambda b, t: 1 - np.exp(-b*t)
        inf_num = np.random.binomial(self.S0, inf_prob(p, d))
        return inf_num

class DeathModelMultiple(Simulator):


    """
    Class to simulate sequential data according to the death model. Used in (non-myopic) cases where population observations are needed at several design times.
    """

    def __init__(self, truth, S0):

        """
        truth: ground truth, scalar
        S0: starting population of death model
        """

        super(DeathModelMultiple, self).__init__(truth)
        self.S0 = S0

    def summary(self, Y):

        Y_psi = list()
        ind = 0
        for arr in Y:
            if len(arr) == 1:
                Y_psi.append([arr[0], 0])
            else:
                Y_psi.append(arr)
            ind += 1
        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        d: design times at which to simulate data (multi-dimensional array); times need to be chronological
        p: model parameter (scalar)
        """

        inf_prob = lambda b, t: 1 - np.exp(-b*t)

        infected = list()
        d0 = 0
        I0 = 0
        if isinstance(d,float):
            inf_num = np.random.binomial(self.S0 - I0,inf_prob(p, d - d0))
            infected.append(inf_num)
        else:
            for idx in range(len(d)):
                if d[idx] < d0:
                    raise ValueError("You can't go backwards in time!")
                inf_num = I0 + np.random.binomial(self.S0-I0, inf_prob(p, d[idx]-d0))
                infected.append(inf_num)
                d0 = d[idx]
                I0 = inf_num

        return np.array(infected)

class SIRModel(Simulator):

    """
    Class to simulate data according to the Susceptible-Infected-Recovered (SIR) model.
    """

    def __init__(self, truth, N):

        """
        truth: ground truth, two-dimensional array
        N: starting (total population), scalar
        """

        super(SIRModel, self).__init__(truth)
        self.N = N
        self.S0 = N - 1
        self.I0 = 1
        self.R0 = 0

    def summary(self, Y):

        Y_psi = list()
        for arr in Y:
            Y_psi.append([arr[0], arr[1], arr[2]])
            #if np.array(arr).shape == ():
            #    tmp = [arr, 0]
                #tmp = [arr ** i for i in range(1,5)]
            #else:
            #    tmp = [arr[0], 0]
                #tmp = [arr[0] ** i for i in range(1, 5)]
            #Y_psi.append(tmp)
        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        d: design variable (scalar)
        p: model parameters (two-dimensional array)
        """

        dt = 0.01
        times = np.arange(0 + dt, d + dt, dt)

        S = self.S0
        I = self.I0
        R = self.R0

        for _ in times:

            pinf = p[0] * I / self.N
            dI = np.random.binomial(S, pinf)

            precov = p[1]
            dR = np.random.binomial(I, precov)

            S = S - dI
            I = I + dI - dR
            R = R + dR

        return np.array([S, I, R])

class SIRModelMultiple(Simulator):

    """
    Class to simulate sequential data according to the Susceptible-Infected-Recovered (SIR) model. Used in (non-myopic) cases where population observations are needed at several design times.
    """

    def __init__(self, truth, N):

        """
        truth: ground truth, two-dimensional array
        N: starting (total population), scalar
        """

        super(SIRModelMultiple, self).__init__(truth)
        self.N = N
        self.S0 = N - 1
        self.I0 = 1
        self.R0 = 0

    def summary(self, Y):

        Y_psi = list()
        for arr in Y:

            #Y_psi.append(arr)

            flat = arr.flatten()
            Y_psi.append(flat)

        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        d: design times at which to simulate data (multi-dimensional array); times need to be chronological
        p: model parameters (two-dimensional array)
        """

        dt = 0.01

        data = list()

        St = self.S0
        It = self.I0
        Rt = self.R0

        for tau in d:

            times = np.arange(0 + dt, tau + dt, dt)

            S = St
            I = It
            R = Rt

            for _ in times:

                pinf = p[0] * I / self.N
                #print(pinf)
                dI = np.random.binomial(S, pinf)

                precov = p[1]
                dR = np.random.binomial(I, precov)

                S = S - dI
                I = I + dI - dR
                R = R + dR

            y = [S, I, R]
            data.append(y)

            St = S
            It = I
            Rt = R

        return np.array(data)
